import streamlit as st
import tensorflow as tf

model = load_model('emotion_model1.h5')

st.write("""
# Mood Classifier App"""
)
file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])

