# Clasificador de Tejidos Cadavéricos con MobileNetV2 y Gradio

Prototipo experimental de visión computacional que clasifica imágenes de laboratorio anatómico en tres categorías de tejido: **huesos**, **músculos** y **venas/arterias**.

El proyecto implementa **Transfer Learning** utilizando una red neuronal convolucional liviana (**MobileNetV2**) preentrenada en ImageNet, adaptando su capa clasificadora final a un conjunto de datos propio mediante PyTorch y exponiendo la inferencia a través de una interfaz gráfica interactiva con Gradio.

---

## Requisitos

- Python 3.10 o superior
- Gestor de paquetes y entornos [uv](https://docs.astral.sh/uv/)
- Conexión a internet (únicamente para la descarga inicial de los pesos del modelo preentrenado)

---

## Instalación en Windows (PowerShell)

1. Clonar el repositorio y posicionarse en la carpeta:
   ```powershell
   git clone [https://github.com/TU_USUARIO/clasificador-anatomia.git](https://github.com/TU_USUARIO/clasificador-anatomia.git)
   cd clasificador-anatomia