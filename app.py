import streamlit as st
import tensorflow as tf

@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('emotion_model1.h5')
  return model
model=load_model()
st.write("""
# Weather Detection System"""
)
file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])
