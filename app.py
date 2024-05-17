import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageOps 
from tensorflow import keras
from keras.models import load_model

st.write("""
# Mood Classifier
""")

file = st.file_uploader("Choose photo from the computer", type=["jpg", "png"])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if file is not None:
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    st.image(image, channels="BGR", caption='Original Image')

    model = load_model('model (1).h5')






    
