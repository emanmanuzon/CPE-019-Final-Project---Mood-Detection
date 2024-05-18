import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_data(experimental_allow_widgets=True)
def load_model():
    model = tf.keras.models.load_model('moodmodel.h5')
    return model

model = load_model()

st.title("Mood Classifier")

st.write("😠 Angry", "😃 Happy", "😐 Neutral", "😢 Sad")

option = st.selectbox(
    'How would you like to provide the image?',
    ('Upload a file', 'Take a picture')
)

file = None

if option == 'Upload a file':
    file = st.file_uploader("Choose an image file", type=["jpg", "png"])
elif option == 'Take a picture':
    file = st.camera_input("Take a picture")

if file is None:
    st.text("Please upload an image file or take a picture")
else:
    st.text("Image successfully loaded!")
    if st.button("Detect Faces and Predict Mood"):
        if option == 'Upload a file':
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
        elif option == 'Take a picture':
            image = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), -1)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        mood_labels = ['Angry', 'Happy', 'Neutral', 'Sad']
        detected_faces = []

        if len(faces) == 0:
            st.image(image, channels="BGR", caption='Original Image')
            st.markdown("<p style='font-size: 30px; color: red;'>No faces detected in the image.</p>", unsafe_allow_html=True)
        else:
            for (x, y, w, h) in faces:
                roi = image[y:y + h, x:x + w]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert to RGB
                roi_pil = Image.fromarray(roi)
                roi_resized = ImageOps.fit(roi_pil, (48, 48))
                roi_array = np.asarray(roi_resized)
                roi_reshaped = roi_array[np.newaxis, ...]

                prediction = model.predict(roi_reshaped)
                label = mood_labels[prediction.argmax()]

                detected_faces.append(label)

                label_position = (x, y - 10)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            st.image(image, channels="BGR", caption=f'Image with {len(faces)} face(s) detected and labeled')

            st.write("Summary of Detected Faces:")
            mood_counts = {mood: detected_faces.count(mood) for mood in mood_labels}
            st.table(mood_counts)
