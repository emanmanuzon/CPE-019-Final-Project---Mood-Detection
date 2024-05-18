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
file=st.file_uploader("Choose photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
if file is None:
    st.text("Please upload an image file")
else:
    if st.button("Detect Faces and Predict Mood"):
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)

      st.image(image, channels="BGR", caption='Original Image')

      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

      mood_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
    
      for (x, y, w, h) in faces:
        roi = image[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert to RGB
        roi_pil = Image.fromarray(roi)
        roi_resized = ImageOps.fit(roi_pil, (48, 48))
        roi_array = np.asarray(roi_resized)
        roi_reshaped = roi_array[np.newaxis, ...]

        prediction = model.predict(roi_reshaped)
        string=np.argmax(prediction)
        st.success(string)
        label = mood_labels[prediction.argmax()]

        label_position = (x, y - 10)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

      st.image(image, channels="BGR", caption=f'Image with {len(faces)} face(s) detected and labeled')
