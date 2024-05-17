import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageOps 
from tensorflow import keras
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import img_to_array

@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('model (1).h5')
  return model
st.write("""
# Mood Classifier"""
)
file=st.file_uploader("Choose photo from the computer",type=["jpg","png"])

if file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)  # Use np.frombuffer to handle binary data

    # Display the original image
    st.image(image, channels="BGR", caption='Original Image')

    # Detect faces in the image
    image_with_faces, num_faces, rois = detect_faces(image)

    # Display the image with detected faces
    st.image(image_with_faces, channels="BGR", caption=f'Image with {num_faces} face(s) detected')

    # Display the ROIs of the detected faces
    for i, roi in enumerate(rois):
        st.image(roi, channels="BGR", caption=f'Region of Interest {i+1}')
