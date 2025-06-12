import streamlit as st
import torch
import numpy as np
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Streamlit UI
st.title("Animal Detector using YOLOv5")
st.write("Upload an image, and the model will detect animals.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting...")
    
    # Perform inference
    results = model(image)
    results.render()
    
    # Convert results to image
    detected_image = Image.fromarray(results.ims[0])
    st.image(detected_image, caption="Detected Objects", use_column_width=True)
    
    st.write("## Detected Objects:")
    for obj in results.pandas().xyxy[0]['name']:
        st.write(f"- {obj}")
