import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Charger le modèle
model = load_model('model.h5', compile=False)

# Liste des classes
class_names = ['F', 'N', 'Q', 'S', 'V']

# Interface Streamlit
st.title("Classification ECG")
uploaded_file = st.file_uploader("Charger une image d'ECG", type=["png", "jpg"])

if uploaded_file is not None:
    # Charger et prétraiter l'image
    image = load_img(uploaded_file, target_size=(128, 128))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)

    # Effectuer la prédiction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)

    # Afficher la classe prédite
    predicted_class_name = class_names[predicted_class]
    st.write("Classe prédite :", predicted_class_name)

    # Afficher l'image chargée
    st.image(image, caption='Image chargée', use_column_width=True)
