import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load your pre-trained model, classes, kcal_values, and age_categories here
# Example placeholders (replace with your actual values)
model = keras.models.load_model('models/Model.h5')
classes = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']
kcal_values = {'burger': 300, 'butter_naan': 150, 'chai': 50, 'chapati': 120, 'chole_bhature': 250, 'dal_makhani': 180, 'dhokla': 80, 'fried_rice': 200, 'idli': 40, 'jalebi': 180, 'kaathi_rolls': 320, 'kadai_paneer': 280, 'kulfi': 150, 'masala_dosa': 180, 'momos': 100, 'paani_puri': 30, 'pakode': 120, 'pav_bhaji': 250, 'pizza': 280, 'samosa': 120}
age_categories = {'Teenagers': ['kulfi', 'burger', 'pizza', 'kaathi_rolls', 'kadai_paneer', 'momos', 'butter_naan', 'dhokla', 'chapati', 'idli', 'jalebi', 'masala_dosa', 'paani_puri', 'pakode'], 'Adults': ['samosa', 'pav_bhaji', 'chole_bhature', 'dal_makhani', 'pakode', 'chai', 'burger', 'butter_naan', 'dhokla', 'chapati', 'fried_rice', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'paani_puri', 'pizza'], 'Children': ['kulfi', 'idli', 'jalebi', 'pakode', 'dhokla']}

def predict(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.xception.preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = classes[predicted_class_index]

    # Get the kcal value for the predicted class
    predicted_kcal_value = kcal_values.get(predicted_class_label, 'N/A')

    # Find the age categories for the predicted class
    predicted_age_categories = []
    for age_cat, food_items in age_categories.items():
        if predicted_class_label in food_items:
            predicted_age_categories.append(age_cat)

    return predicted_class_label, predicted_kcal_value, predicted_age_categories

# Streamlit app
st.title("Food Recognition and Calorie Estimation App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    predicted_class, predicted_kcal, predicted_age_cats = predict(uploaded_file)

    # Display the results
    st.subheader("Prediction Results:")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Kcal Value: {predicted_kcal} kcal")
    st.write(f"Age Categories: {', '.join(predicted_age_cats)}")
