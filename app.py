import streamlit as st
import tensorflow as tf
from keras.models import load_model

@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('model (1).h5')
  return model
st.write("""
# Mood Classifier"""
)
file=st.file_uploader("Choose photo from the computer",type=["jpg","png"])

