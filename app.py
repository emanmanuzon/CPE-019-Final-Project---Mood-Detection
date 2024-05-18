import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam
from PIL import Image


model=tf.keras.models.load_model('CNN_Model_7.h5')
model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])
st.write("""
# Mood Classifier"""
)

file = st.file_uploader("Choose photo from the computer", type=["jpg", "png"])
