import streamlit as st
import tensorflow as tf
from keras.models import load_model

model = load_model('model (1).h5')
st.write("""
# Mood Classifier"""
)
file=st.file_uploader("Choose photo from the compu",type=["jpg","png"])

