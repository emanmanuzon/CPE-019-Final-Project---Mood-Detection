import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

def load_keras_model():
    model = load_model('model (1).h5')
    return model
model = load_keras_model()
st.write("""
# Mood Classifier"""
)

file = st.file_uploader("Choose photo from the computer", type=["jpg", "png"])
