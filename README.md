[README_spline.md](https://github.com/user-attachments/files/33110455/README_spline.md)
# Silueta de imágenes con interpolación B-Spline

Programa en Python que toma una imagen, detecta automáticamente el borde del objeto y lo reconstruye como una **curva B-Spline cúbica cerrada** que pasa por un conjunto de nodos ubicados sobre el contorno. También detecta los **huecos internos** (por ejemplo, los ojos de un personaje) y los traza con su propia spline.

Proyecto de práctica para la materia de **Métodos Numéricos**, tema: *interpolación por splines*.

![Comparación entre la imagen original y la silueta con B-Spline](imagenes/resultado.png)

> Agrega aquí una captura de la ventana del programa (imagen original a la izquierda y la silueta a la derecha) en `imagenes/resultado.png`.

---

## ¿Qué hace?

1. Lee una imagen.
2. Separa el objeto del fondo.
3. Extrae el contorno exterior de cada pieza y los contornos de sus huecos.
4. Reparte **nodos** equiespaciados a lo largo de cada contorno.
5. Ajusta una **B-Spline cúbica cerrada** por cada contorno.
6. Muestra en una sola ventana la imagen original y la silueta reconstruida, con ejes compartidos (el zoom de un panel se refleja en el otro).

| Elemento | Color |
|---|---|
| Relleno del objeto | cian |
| Curva B-Spline del borde exterior | azul |
| Nodos del borde exterior | rojo |
| Huecos (ojos, agujeros) | blanco, con nodos magenta |

---

## Fundamento matemático

Una **spline** reemplaza un único polinomio de grado alto por polinomios de grado bajo definidos por tramos, unidos en los nodos. Así se evita la oscilación de la interpolación polinomial con muchos puntos (fenómeno de Runge).

En una spline cúbica, cada tramo es un polinomio de grado 3:

$$S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3$$

y en los nodos interiores se exige:

- que la curva pase por los puntos dados,
- continuidad de la primera derivada,
- continuidad de la segunda derivada (continuidad $C^2$).

Aplicado a una imagen, no se interpola una función $y = f(x)$ sino una **curva paramétrica** $(x(t), y(t))$ con $t \in [0,1]$, es decir, una spline para cada coordenada. Como el contorno de un objeto es una curva cerrada, se usa una spline **periódica**.

Puntos que conviene tener claros:

- La suavidad la dan el grado y las condiciones de continuidad, no la cantidad de nodos.
- Más nodos hacen que la curva se parezca más al contorno real: para funciones suficientemente suaves, el error de interpolación del spline cúbico disminuye como $O(h^4)$, donde $h$ es la separación entre nodos.
- Los valores intermedios no se inventan: se **evalúan** en la spline una vez construida (`splev`).

---

## Requisitos

- Python 3.9 o superior
- Librerías:

| Librería | Uso |
|---|---|
| `opencv-python` | lectura de la imagen y detección de contornos |
| `scipy` | construcción y evaluación de la B-Spline (`splprep`, `splev`) |
| `matplotlib` | gráficos |
| `numpy` | manejo de arreglos |

## Instalación

```bash
git clone https://github.com/Novaleon630/NOMBRE-DEL-REPOSITORIO.git
cd NOMBRE-DEL-REPOSITORIO
pip install opencv-python scipy matplotlib numpy
```

> Reemplaza `NOMBRE-DEL-REPOSITORIO` por el nombre real de tu repositorio.

---

## Uso

1. Guarda tu imagen en la misma carpeta que el script.
2. Ajusta la variable `RUTA_IMAGEN` al inicio de `silueta_comparar.py`:

   ```python
   RUTA_IMAGEN = r"caballerito2.jpg"
   # o una ruta completa en Windows (ojo con la r delante de las comillas):
   # RUTA_IMAGEN = r"C:\Users\TuUsuario\Desktop\caballerito2.jpg"
   ```

3. Ejecuta:

   ```bash
   python silueta_comparar.py
   ```

   o pasando la imagen como argumento:

   ```bash
   python silueta_comparar.py otra_imagen.png
   ```

### Parámetros de configuración

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `RUTA_IMAGEN` | `"caballerito2.jpg"` | Imagen a procesar. |
| `N_NODOS_POR_1000_PX` | `120` | Densidad de nodos por cada 1000 px de perímetro. Más nodos siguen mejor el borde. |
| `AREA_MINIMA` | `0.005` | Se ignoran piezas menores a este porcentaje de la pieza más grande. |
| `AREA_MINIMA_HUECO` | `0.0005` | Igual, pero para huecos. Súbelo (por ejemplo `0.005`) para quedarte solo con los huecos grandes, como los ojos. |
| `SUAVIDAD` | `0` | `0` = la curva pasa exactamente por los nodos (interpolación). Valores mayores suavizan y aproximan. |
| `REFINAR_CON_GRABCUT` | `False` | `True` aplica GrabCut para afinar la máscara. Es más lento. |
| `SUPERPONER_EN_IMAGEN` | `False` | `True` dibuja además la curva sobre la imagen original. |

---

## Cómo funciona el código

| Función | Qué hace |
|---|---|
| `mascara_por_fondo` | Estima el color del fondo a partir de los bordes de la imagen. Combina la distancia a ese color con un umbral de Otsu sobre la imagen en grises (con CLAHE) para obtener la máscara del objeto. |
| `obtener_contornos` | Usa `cv2.findContours` con `RETR_CCOMP`, que organiza los contornos en dos niveles: bordes exteriores y huecos. Filtra por área. |
| `muestrear_por_arco` | Reparte los nodos a distancias iguales **a lo largo del borde** (longitud de arco), no por posición en la lista de píxeles. |
| `bspline_cerrada` | Ajusta la B-Spline cúbica periódica con `splprep` y la evalúa con `splev`. |
| `dibujar_silueta` | Dibuja relleno, curvas y nodos. |

### Detalle importante sobre `splprep`

Con `per=True`, `splprep` **ignora el último punto de la lista**. Por eso el código repite el primer nodo al final. Sin esto, el último nodo real queda fuera de la curva y la interpolación deja de ser exacta en ese punto.

---

## Estructura sugerida del repositorio

```
.
├── README.md
├── silueta_comparar.py      # versión principal: detección automática + huecos + vista comparativa
├── imagenes/
│   ├── caballerito2.jpg     # imagen de ejemplo
│   └── resultado.png        # captura para el README
└── ...                      # otras versiones o pruebas del trabajo
```

---

## Limitaciones

- Funciona mejor con imágenes de **buen contraste y fondo liso** (siluetas, dibujos en línea, objetos sobre fondo blanco). Con fondos con degradados, brillos o colores parecidos al objeto, la máscara automática puede mezclar fondo y objeto.
- `AREA_MINIMA_HUECO` depende de cada imagen. Un valor muy bajo detecta detalles pequeños (grietas, texturas) como huecos.
- Las zonas muy delgadas (colas, colmillos) necesitan más nodos para no deformarse.
- Un hueco que contiene una pieza dentro de él se trata como pieza independiente (por cómo `RETR_CCOMP` organiza la jerarquía).

---

## Posibles mejoras

- Selección manual de nodos sobre la imagen de fondo ("calco") para colocarlos donde la curva cambia de dirección.
- Exportar los nodos a CSV o los contornos a SVG.
- Comparar numéricamente el error entre la spline y el contorno original según el número de nodos.

---

## Autor

**Nivardo Román León Dueñas** · [@Novaleon630](https://github.com/Novaleon630)
Universidad Mayor de San Andrés (UMSA), La Paz, Bolivia.
