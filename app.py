import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('CNN_Model_7.h5')
  return model
model=load_model()

st.write("""
# Mood Classifier"""
)

file = st.file_uploader("Choose photo from the computer", type=["jpg", "png"])
