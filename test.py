import streamlit as st
import cv2
import numpy as np

st.title("OpenCV Test")
img = np.zeros((200,200,3), dtype=np.uint8)
cv2.putText(img, "CV2 works!", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
st.image(img)
