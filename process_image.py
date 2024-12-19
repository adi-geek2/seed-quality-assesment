# Seed quality assessment using MobileNetV3
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import warnings

# Suppress warnings related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to load the pre-trained model
def load_model():
    model = models.mobilenet_v3_large()
    num_classes = 2
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenet_weights_20241129_132949.pth", map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model


# Preprocessing pipeline for images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(image_path).convert('RGB')  # Open and convert to RGB
    return transform(image).unsqueeze(0)  # Add batch dimension


# Main function
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]  # Get image path from PHP script
    try:
        # Preprocess the image
        input_tensor = preprocess_image(image_path)

        # Load model and predict
        model = load_model()
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Map prediction to class labels
        class_labels = {0: "Bad Seed", 1: "Good Seed"}

        # Return result as JSON
        result = {
            "quality": class_labels[predicted_class],
        }
        print(json.dumps(result))
    except Exception as e:
        # Handle any errors
        print(json.dumps({"error": str(e)}))
