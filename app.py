import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageOps 
from tensorflow import keras
from keras.models import load_model

@st.cache_data(experimental_allow_widgets=True)
@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('model (1).h5')
  return model
model=load_model()

st.write("""
# Mood Classifier
""")

file = st.file_uploader("Choose photo from the computer", type=["jpg", "png"])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if file is not None:
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_data=Image.open(file)
    st.image(image_data, channels="BGR", caption='Original Image')
    size=(100,100)
    image=ImageOps.fit(image_data,size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)








    
