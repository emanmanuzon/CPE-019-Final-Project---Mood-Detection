import streamlit as st
import tensorflow as tf

@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('moodmodel.h5')
  return model
model=load_model()
st.write("""
# Mood Classifier"""
)
file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    size=(48,48)
    image=ImageOps.fit(image,size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    class_names=['Angry', 'Happy', 'Neutral', 'Sad']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
