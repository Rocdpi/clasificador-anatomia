import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import pillow_heif
import gradio as gr

# Habilitar soporte para fotos HEIC en Pillow
pillow_heif.register_heif_opener()

# 1. Configuración de dispositivo y transformaciones
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Carga del dataset local reconociendo .heic
data_dir = "data"
if not os.path.exists(data_dir):
    raise FileNotFoundError("No se encontró la carpeta 'data/'.")

valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.heic', '.HEIC')

def is_valid_image(path):
    return path.lower().endswith(valid_extensions)

dataset = datasets.ImageFolder(root=data_dir, transform=transform_train, is_valid_file=is_valid_image)
classes = dataset.classes  # Detecta ['huesos', 'musculos', 'venas arterias']
print(f"Clases detectadas: {classes}")
print(f"Total de imágenes cargadas: {len(dataset)}")

loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. Transfer Learning con MobileNetV2
print("Configurando el modelo preentrenado...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.001)

print("Ajustando la capa final del clasificador...")
model.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Época {epoch + 1}/5 - Pérdida: {running_loss / len(loader):.4f}")

model.eval()
print("Modelo listo. Iniciando interfaz...")

# 4. Inferencia
def clasificar_imagen(imagen_pil):
    if imagen_pil is None:
        return {}
    
    # Asegurar modo RGB (algunas fotos de celular vienen en RGBA o formatos especiales)
    if imagen_pil.mode != "RGB":
        imagen_pil = imagen_pil.convert("RGB")
        
    img_tensor = transform_infer(imagen_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    return {classes[i]: float(probabilities[i]) for i in range(len(classes))}

# 5. Interfaz Gradio
demo = gr.Interface(
    fn=clasificar_imagen,
    inputs=gr.Image(type="pil", label="Cargar fotografía anatómica"),
    outputs=gr.Label(num_top_classes=len(classes), label="Probabilidades de clasificación"),
    title="Clasificador Anatómico de Tejidos Cadavéricos",
    description="Prototipo experimental con Transfer Learning (MobileNetV2) para clasificar: huesos, músculos y venas/arterias."
)

if __name__ == "__main__":
    demo.launch()