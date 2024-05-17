import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

def load_keras_model():
    model = load_model('model.h5')
    return model
model = load_keras_model()
st.write("""
# Weather Detection System"""
)
file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])

image = Image.open(file)
image = image.resize((48, 48))  # Resize to the input size the model expects
image = img_to_array(image)  # Convert the image to an array
image = np.expand_dims(image, axis=0)  # Add batch dimension
image = image / 255.0  # Normalize the image if the model expects normalized input
predictions = model.predict(image)
