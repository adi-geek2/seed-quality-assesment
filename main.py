"""
# Seed quality assesment app
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sqlite3


@st.cache_resource
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

# Database connection
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            image BLOB,
            prediction TEXT
        )
    """)
    return conn

# Streamlit App
st.title("Deep Learning with PyTorch - Streamlit Integration")

# Upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Load model and predict
    model = load_model()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Display prediction
    st.write(f"Prediction: Class {predicted_class}")

    # Save data to the database
    conn = get_db_connection()
    conn.execute("INSERT INTO predictions (image, prediction) VALUES (?, ?)", 
                 [uploaded_file.getvalue(), str(predicted_class)])
    conn.commit()
    conn.close()

    st.success("Prediction saved to the database!")
