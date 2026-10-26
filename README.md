# 🐕 Detector de puntos clave en perros

Demo de **visión por computador** que localiza **24 puntos anatómicos** de un perro (patas, orejas, ojos, nariz,
cola…) sobre una foto y dibuja un **contorno azul** que bordea la silueta del animal.

Hecho con **Keras 3** (MobileNetV2 + cabeza de **mapas de calor**) y una app de **Streamlit** lista para correr en
local. El modelo se entrena y se valida en [`cuaderno_entrenamiento.ipynb`](cuaderno_entrenamiento.ipynb).

![App](docs/preview.jpg)

> ⚠️ **Es una demostración del pipeline, no un detector de producción.** El modelo se entrenó con la muestra pública
> de StanfordExtra: **25 imágenes de 4 razas**. Acierta en perros parecidos a los de la muestra y falla con perros
> muy distintos. Ver [Limitaciones](#️-limitaciones).

---

## ✨ Qué hace

| | |
|---|---|
| **Entrada** | Una foto de un perro (JPG/PNG) o una de las 3 muestras incluidas |
| **Salida** | 24 puntos con su color, el contorno de la silueta y la tabla de coordenadas |
| **Formato** | Coordenadas en píxeles y normalizadas `[0, 1]`, confianza por punto, descarga en CSV y PNG |

El contorno **no es una máscara de segmentación**: es la **envolvente convexa de los puntos que el modelo considera
visibles**, suavizada con el algoritmo de Chaikin.

## 🚀 Instalación y uso

```bash
git clone https://github.com/Joserraldo/detector-puntos-clave-perro.git
cd detector-puntos-clave-perro
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`. En la barra lateral puedes elegir una muestra o subir tu propia foto, activar o
desactivar el contorno, numerar los puntos y ajustar grosores. El modelo (22 MB) ya viene en `modelos/`: no hace
falta entrenar nada para probarlo.

## 📊 Precisión (medida en imágenes que el modelo NO vio)

Validación cruzada de 5 pliegues: se entrena 5 veces dejando fuera 5 imágenes distintas y el error se mide solo en
esas. Error = distancia media entre el punto predicho y el real, en % del tamaño de la imagen (puntos anotados).

| | Error en imágenes no vistas |
|---|---|
| Referencia: predecir siempre la pose media | **19,1 %** |
| **Este modelo** | **8,7 %** (por pliegue: 10,7 · 7,7 · 8,0 · 10,3 · 6,9) |
| Sobre las propias imágenes de entrenamiento (no es generalización) | 2,2 % |

Acierto de visibilidad: 77 %. Detalle en [`resultados/validacion_cruzada.json`](resultados/validacion_cruzada.json).

Predicciones sobre imágenes **no vistas** (aro verde = anotación real, punto = predicción):

![Predicciones](docs/predicciones.jpg)

## 🧠 Cómo funciona

```
Imagen 224×224×3 (uint8)
  → preprocess_input (MobileNetV2)
  → MobileNetV2 · ImageNet           (último bloque entrenable, BatchNorm congelado)   → 7×7×1280
  → 3 × [Conv2DTranspose(128, 4, stride 2) → BatchNorm → ReLU]                         → 56×56×128
  → Conv2D 1×1 (24)                                                                    → 56×56×24
```

1. **Transfer learning:** se reutiliza MobileNetV2 preentrenado en ImageNet.
2. **Mapas de calor:** la red devuelve un mapa de 56×56 por punto. La posición es el pico del mapa (refinada con un
   centroide local) y la **altura del pico** es la confianza de que el punto se vea. Los puntos con confianza baja
   (umbral 0,19, calibrado por validación cruzada) se dibujan como anillos tenues y no entran en el contorno.
3. **Por qué no regresión directa:** una primera versión regresaba 48 coordenadas tras un pooling global. Con 25
   imágenes no localizaba: en imágenes no vistas su error (≈ 20 %) era igual al de predecir la pose media.
4. **Correcciones de datos:** el volteo horizontal intercambia las etiquetas izquierda/derecha, los puntos no anotados
   no se entrenan como si estuvieran en (0, 0) y un punto que sale del cuadro por el aumento deja de contar como
   visible.
5. **Dibujo:** todo con Pillow a 3× de resolución y reducido al final (anti-aliasing), sin OpenCV.

La muestra **nunca anota** los puntos 20-23 (ojo izquierdo, ojo derecho, cruz y garganta), así que la app los dibuja
como anillos tenues.

## 🔁 Reentrenar

* **Cuaderno** (recomendado): [`cuaderno_entrenamiento.ipynb`](cuaderno_entrenamiento.ipynb), pensado para **Google
  Colab con GPU**. Descarga la muestra de StanfordExtra, hace la validación cruzada y guarda el modelo.
* **Script equivalente:** `python entrenar_modelo.py` (`--rapido` para una prueba de humo). Necesita la carpeta
  `StanfordExtra-master/` (la descarga el cuaderno, o clona [benjiebob/StanfordExtra](https://github.com/benjiebob/StanfordExtra)).
* **GPU:** TensorFlow en Windows nativo no usa la GPU. Con PyTorch instalado (con CUDA) el cuaderno y el script
  entrenan en ella automáticamente (Keras 3 usa el backend de PyTorch) y el `.keras` resultante sirve con cualquier
  backend. Los 5 pliegues + modelo final tardan ≈ 15 min en una RTX 4060 y ≈ 1 h en CPU.

```bash
python test_prediccion.py   # el modelo carga y los puntos no colapsan (código de salida 1 si falla)
python test_app.py          # prueba de extremo a extremo de la app
python evaluar_modelo.py    # figuras sobre las 25 imágenes de entrenamiento (no mide generalización)
```

## 📁 Estructura

```
detector-puntos-clave-perro/
├── app.py                       # interfaz Streamlit
├── puntos_clave.py              # inferencia, contorno y dibujo (reutilizable)
├── cuaderno_entrenamiento.ipynb # entrenamiento + validación cruzada (ejecutado)
├── entrenar_modelo.py           # el mismo entrenamiento por línea de comandos
├── evaluar_modelo.py · test_prediccion.py · test_app.py
├── modelos/modelo_puntos_clave.keras
├── muestras/                    # 3 fotos de ejemplo
├── resultados/                  # validación cruzada y figuras
├── docs/                        # capturas de este README
├── .streamlit/config.toml       # tema
└── requirements.txt
```

`puntos_clave.py` se puede importar desde cualquier script:

```python
from PIL import Image
from puntos_clave import cargar_modelo, predecir, dibujar

modelo = cargar_modelo("modelos/modelo_puntos_clave.keras")
img = Image.open("muestras/rhodesian_1.jpg").convert("RGB")

kps, vis = predecir(modelo, img)     # (24, 2) en [0,1]  y  (24,) de confianza de visibilidad
dibujar(img, kps, vis=vis).save("salida.png")
```

## ⚠️ Limitaciones

* **25 imágenes de 4 razas** (Rhodesian ridgeback, Weimaraner, Border terrier, Golden retriever). Cada pliegue de la
  validación tiene solo 5 imágenes, así que el 8,7 % es una estimación ruidosa.
* Perros muy distintos, cachorros, poses raras o varios perros en la foto pueden fallar.
* Para uso real hay que entrenar con el StanfordExtra completo (≈ 12 000 imágenes, se solicita a los autores): basta
  apuntar `JSON_PATH` del cuaderno al JSON completo y ejecutarlo en Colab con GPU.

## 📚 Créditos

- **Dataset:** [StanfordExtra](https://github.com/benjiebob/StanfordExtra) — B. Biggs et al., *Who Left the Dogs Out?
  3D Animal Reconstruction with Expectation Maximization in the Loop* (ECCV 2020). Imágenes del
  [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).
- **Modelo base:** MobileNetV2 (Sandler et al., 2018) vía `keras.applications`.
- **Idea del cuaderno:** *Detección de puntos clave con aprendizaje por transferencia* (Alfredo Díaz Claro, sobre el
  trabajo de Sayak Paul y Muhammad Anas Raza).
- Código, app y reentrenamiento: **José Alejandro Téllez Prada**.

## 📄 Licencia

MIT — ver [LICENSE](LICENSE). El dataset StanfordExtra tiene su propia licencia (uso académico); revisa su
repositorio antes de redistribuir imágenes.
