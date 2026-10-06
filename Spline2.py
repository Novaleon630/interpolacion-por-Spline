"""
Silueta AUTOMÁTICA con B-Spline (versión final pulida)
------------------------------------------------------
Objetivo: bordes suaves, muchos nodos intermedios, contornos detallados.

CAMBIOS vs versión anterior:
  1. NODOS MUCHO MÁS DENSOS: subimos N_NODOS_POR_1000_PX de 120 a 300.
  2. INTERPOLACIÓN DENTRO DE CADA TRAMO: la curva se evalúa en 5000 puntos
     (antes 2000) → curvas más lisas visualmente.
  3. SUBDIVISIÓN PREVIA DEL CONTORNO: se insertan puntos intermedios con
     interpolación lineal ANTES de ajustar la B-Spline. Resultado: la curva
     pasa más pegada al borde real, sin "redondear" esquinas.
  4. PRE-SUAVIZADO DEL CONTORNO (opcional): filtro de media móvil circular
     sobre las coordenadas x,y del contorno para quitar dientes de sierra.
  5. GRABCUT ACTIVADO por defecto → bordes más limpios en zonas grises.
  6. Grado de la B-Spline configurable (k=3 cúbica, k=4 cuártica...).
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d

# ----------------------------------------------------------------------
# CONFIGURACIÓN  (lo que puedes tocar)
# ----------------------------------------------------------------------
RUTA_IMAGEN = r"bill.jpg"

REFINAR_CON_GRABCUT = True   # True = bordes más limpios (un poco más lento)
N_NODOS_POR_1000_PX = 300    # densidad de nodos sobre el borde (¡subido!)
SUBDIVISION         = 4      # inserta N puntos intermedios entre cada píxel del contorno
SUAVIZADO_CONTORNO  = 3      # ventana del filtro de media móvil (0 = desactivado)
N_PUNTOS_CURVA      = 5000   # puntos con los que se dibuja la B-Spline final
GRADO_BSPLINE       = 3      # 3 = cúbica (suave), 4/5 = más suave aún

AREA_MINIMA         = 0.004  # piezas mayores al 0.4% de la más grande
AREA_MINIMA_HUECO   = 0.0003 # agujeros pequeños (ojos, detalles internos)
SUAVIDAD            = 0      # 0 = interpola exacto los nodos

SUPERPONER_EN_IMAGEN = True  # dibuja también la curva sobre la imagen original
# ----------------------------------------------------------------------


def leer_imagen(ruta):
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No pude abrir '{ruta}'.")
    return img


# ----------------------------------------------------------------------
# 1. Segmentación
# ----------------------------------------------------------------------
def mascara_por_fondo(img):
    h, w = img.shape[:2]
    m = max(2, min(h, w) // 100)
    borde = np.concatenate([
        img[:m].reshape(-1, 3), img[-m:].reshape(-1, 3),
        img[:, :m].reshape(-1, 3), img[:, -m:].reshape(-1, 3),
    ])
    fondo = np.median(borde, axis=0)

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gris = clahe.apply(gris)
    gris = cv2.bilateralFilter(gris, 7, 40, 40)   # preserva bordes mejor que Gaussian

    dist_color = np.linalg.norm(img.astype(float) - fondo, axis=2).astype(np.uint8)
    dist_color = cv2.GaussianBlur(dist_color, (3, 3), 0)

    _, m_color = cv2.threshold(dist_color, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m_gris  = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(borde) > 127:
        m_gris = cv2.bitwise_not(m_gris)

    mascara = cv2.bitwise_or(m_color, m_gris)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def refinar_grabcut(img, mascara):
    k = max(3, min(img.shape[:2]) // 40)
    cerca  = cv2.dilate(mascara, np.ones((k, k), np.uint8))
    dentro = cv2.erode(mascara,  np.ones((k, k), np.uint8))
    gc = np.full(mascara.shape, cv2.GC_BGD, np.uint8)
    gc[cerca  > 0] = cv2.GC_PR_BGD
    gc[mascara> 0] = cv2.GC_PR_FGD
    gc[dentro > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, gc, None, bgd, fgd, 10, cv2.GC_INIT_WITH_MASK)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)


def obtener_contornos(mascara, area_min_objeto, area_min_hueco):
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_NONE)
    if not contornos:
        raise RuntimeError("No se encontró ningún objeto.")
    areas = [cv2.contourArea(c) for c in contornos]
    mayor = max(areas)
    externos, huecos = [], []
    for i, c in enumerate(contornos):
        es_externo = jerarquia[0][i][3] == -1
        if es_externo and areas[i] >= area_min_objeto * mayor:
            externos.append(c[:, 0, :].astype(float))
        elif (not es_externo) and areas[i] >= area_min_hueco * mayor:
            huecos.append(c[:, 0, :].astype(float))
    return externos, huecos


# ----------------------------------------------------------------------
# 2. Pre-procesado del contorno  (NUEVO)
# ----------------------------------------------------------------------
def subdividir_contorno(contorno, factor):
    """
    Inserta `factor` puntos intermedios entre cada par de puntos consecutivos
    del contorno mediante interpolación lineal. Resultado: borde más denso,
    la B-Spline se ajusta con más fidelidad.
    """
    if factor <= 1:
        return contorno
    pts = np.vstack([contorno, contorno[:1]])       # cerramos
    nuevos = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        for t in np.linspace(0, 1, factor, endpoint=False):
            nuevos.append(p0 + t * (p1 - p0))
    return np.asarray(nuevos)


def suavizar_contorno(contorno, ventana):
    """
    Filtro de media móvil CIRCULAR sobre x e y por separado.
    Quita dientes de sierra sin desplazar la curva.
    """
    if ventana < 2:
        return contorno
    x = uniform_filter1d(contorno[:, 0], size=ventana, mode="wrap")
    y = uniform_filter1d(contorno[:, 1], size=ventana, mode="wrap")
    return np.column_stack([x, y])


# ----------------------------------------------------------------------
# 3. B-Spline
# ----------------------------------------------------------------------
def muestrear_por_arco(contorno, n_nodos):
    pts = np.vstack([contorno, contorno[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    objetivo = np.linspace(0, s[-1], n_nodos, endpoint=False)
    return np.column_stack([np.interp(objetivo, s, pts[:, 0]),
                            np.interp(objetivo, s, pts[:, 1])])


def bspline_cerrada(contorno, nodos_por_1000, suavidad, grado):
    # 1) subdividir → más densidad de puntos
    contorno = subdividir_contorno(contorno, SUBDIVISION)
    # 2) suavizar → sin dientes de sierra
    contorno = suavizar_contorno(contorno, SUAVIZADO_CONTORNO)

    # 3) muestrear nodos equiespaciados por arco
    paso = np.hypot(*np.diff(np.vstack([contorno, contorno[:1]]), axis=0).T)
    n_nodos = int(max(12, paso.sum() / 1000 * nodos_por_1000))
    nodos = muestrear_por_arco(contorno, n_nodos)

    # 4) B-Spline periódica de grado `grado`
    x = np.append(nodos[:, 0], nodos[0, 0])
    y = np.append(nodos[:, 1], nodos[0, 1])
    tck, _ = splprep([x, y], s=suavidad, k=grado, per=True)

    # 5) evaluar en MUCHOS puntos intermedios → curva lisa
    xs, ys = splev(np.linspace(0, 1, N_PUNTOS_CURVA), tck)
    return nodos, np.asarray(xs), np.asarray(ys)


# ----------------------------------------------------------------------
# 4. Dibujo
# ----------------------------------------------------------------------
def dibujar_silueta(ax, externos, huecos):
    total = 0
    for c in externos:
        nodos, xs, ys = bspline_cerrada(c, N_NODOS_POR_1000_PX, SUAVIDAD, GRADO_BSPLINE)
        ax.fill(xs, ys, color="cyan", alpha=0.85, zorder=1)
        ax.plot(xs, ys, "b-", linewidth=1.2, zorder=2)
        ax.plot(nodos[:, 0], nodos[:, 1], "ro", markersize=2, zorder=3)
        total += len(nodos)
    for c in huecos:
        nodos, xs, ys = bspline_cerrada(c, N_NODOS_POR_1000_PX, SUAVIDAD, GRADO_BSPLINE)
        ax.fill(xs, ys, color="white", zorder=4)
        ax.plot(xs, ys, "k-", linewidth=1.0, zorder=5)
        ax.plot(nodos[:, 0], nodos[:, 1], "mo", markersize=2, zorder=6)
        total += len(nodos)
    return total


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else RUTA_IMAGEN
    img = leer_imagen(ruta)
    h, w = img.shape[:2]

    mascara = mascara_por_fondo(img)
    if REFINAR_CON_GRABCUT:
        mascara = refinar_grabcut(img, mascara)
    externos, huecos = obtener_contornos(mascara, AREA_MINIMA, AREA_MINIMA_HUECO)

    fig, (ax_img, ax_sp) = plt.subplots(1, 2, figsize=(14, 7),
                                        sharex=True, sharey=True)

    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_img.set_title("Imagen original")

    total = dibujar_silueta(ax_sp, externos, huecos)
    ax_sp.set_xlim(0, w)
    ax_sp.set_ylim(h, 0)
    ax_sp.set_aspect("equal")
    ax_sp.grid(alpha=0.3)
    ax_sp.set_title(f"B-Spline: {total} nodos, {len(externos)} pieza(s), "
                    f"{len(huecos)} hueco(s)")

    if SUPERPONER_EN_IMAGEN:
        for c in externos + huecos:
            _, xs, ys = bspline_cerrada(c, N_NODOS_POR_1000_PX, SUAVIDAD, GRADO_BSPLINE)
            ax_img.plot(xs, ys, "-", color="lime", linewidth=1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()