import streamlit as st
import tensorflow as tf
from keras.models import load_model

model = load_model('CNN_Model_7.h5')
st.write("""
# Weather Detection System"""
)
file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])

