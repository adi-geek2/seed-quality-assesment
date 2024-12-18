"""
# Seed quality assesment using mobilnet v3
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def load_model():
    # Load pre-trained MobileNetV3 large model
    model = models.mobilenet_v3_large()
    num_classes = 2
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenet_weights_20241129_132949.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocessing pipeline for images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Preprocess the image
img = Image.open("uploads\123.jpg").convert('RGB')  # Open image
input_tensor = preprocess_image(img)

# Load model and predict
model = load_model()
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# Display prediction
print(f"Prediction: Class {predicted_class}")