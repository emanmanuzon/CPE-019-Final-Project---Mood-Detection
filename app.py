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

def detect_faces(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces and extract ROIs
    rois = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi = image[y:y+h, x:x+w]
        rois.append(roi)
    
    return image, len(faces), rois

def import_and_predict(image_data, classifier):
    # Convert NumPy arra
    image = Image.fromarray(image_data)
    
    image = image.resize((48,48))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = classifier.predict(img_array)
    
    return prediction

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

    for i, roi in enumerate(rois):
        st.image(roi, channels="BGR", caption=f'Region of Interest {i+1}')
