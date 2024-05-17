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

    

    model = load_model('emotion_model1.h5')

    image_data=Image.open(file)
    st.image(image_data, channels="BGR", caption='Original Image')
    size=(48,48)
    image=ImageOps.fit(image_data,size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)








    
