import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Set Streamlit page config
st.set_page_config(page_title="Vehicle Detection", layout="wide")

st.title("🚗 Vehicle Detection & Segmentation with YOLOv8")
st.write("Upload an image to detect and segment vehicles using a custom-trained YOLOv8 model.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Make sure your model is in the same folder
    return YOLO(model_path)

model = load_model()

# Class names from your custom dataset
class_names = ['person', 'auto-rickshaw', 'zeep', 'tempo', 'toto', 'e-rickshaw',
               'bus', 'car', 'trak', 'cyclist', 'taxi', 'bike', 'van', 'cycle-rickshaw']

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Failed to load image.")
    else:
        st.success("✅ Image uploaded successfully!")

        # Run inference
        with st.spinner("Running YOLOv8 segmentation..."):
            results = model(img)[0]

        # Extract data
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        masks = results.masks.data.cpu().numpy() if results.masks is not None else []

        # Count object occurrences
        object_counter = Counter([class_names[c] for c in classes])

        # Generate random colors
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

        # Prepare image and overlay
        img_vis = img.copy()
        overlay = img.copy()

        for i, box in enumerate(boxes):
            cls_id = classes[i]
            conf = scores[i]
            color = [int(c) for c in colors[cls_id]]
            label = f"{class_names[cls_id]} {conf:.2f}"

            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

            # Label with white background
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_vis, (x1, y1 - th - baseline - 5), (x1 + tw, y1), (255, 255, 255), -1)
            cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if len(masks) > 0:
                mask = masks[i].astype(np.uint8) * 255
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[:, :] = color
                mask_indices = mask_resized > 128
                overlay[mask_indices] = cv2.addWeighted(overlay, 0.5, colored_mask, 0.5, 0)[mask_indices]

        # Blend overlay with the original image
        alpha = 0.6
        blended = cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        # Display with matplotlib
        st.subheader("📸 Detected Image")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(blended_rgb)
        ax.axis("off")
        st.pyplot(fig)

        # Show object counts
        st.subheader("📊 Detected Objects Summary")
        st.write(pd.DataFrame(object_counter.items(), columns=["Class", "Count"]))

