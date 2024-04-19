import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from io import BytesIO
import base64
# Load the trained model
model_path = 'css_model.h5'
model = load_model(model_path)

def convert_to_ascii(sentence):
    sentence_ascii = []

    for i in sentence:
        if ord(i) < 8222:
            if ord(i) == 8217:
                sentence_ascii.append(134)
            if ord(i) == 8221:
                sentence_ascii.append(129)
            if ord(i) == 8220:
                sentence_ascii.append(130)
            if ord(i) == 8216:
                sentence_ascii.append(131)
            if ord(i) == 8217:
                sentence_ascii.append(132)
            if ord(i) == 8211:
                sentence_ascii.append(133)

            if ord(i) <= 128:
                sentence_ascii.append(ord(i))

    zer = np.zeros((10000))
    for i in range(len(sentence_ascii)):
        zer[i] = sentence_ascii[i]

    zer.shape = (100, 100)
    return zer

# Function to preprocess new data
def preprocess_new_data(sentence):
    image = convert_to_ascii(sentence)
    x = np.asarray(image, dtype='float')
    image = cv2.resize(x, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    image /= 128
    return image.reshape(1, 100, 100, 1)
# Function to encode image to base64
def image_to_base64(image):
    image_pil = Image.fromarray(np.uint8(image))
    image_buffer = BytesIO()
    image_pil.save(image_buffer, format="JPEG")
    return base64.b64encode(image_buffer.getvalue()).decode()

# Home page
def home():
    
    st.title("XSS Detection App")
    st.write("Welcome to the XSS Detection App! This app is designed to identify potential Cross-Site Scripting (XSS) attacks in input sentences.")
    
    st.image("xss.jpg", use_column_width=True)

    st.header("How to Use the App")
    st.write("1. Navigate to the 'Prediction' page using the menu on the left.")
    st.write("2. Enter a sentence in the text input box.")
    st.write("3. The app will analyze the input using a machine learning model trained to detect XSS attacks.")
    st.write("4. The prediction will be displayed, indicating whether the input contains an XSS attack or is benign.")
    
    st.header("About the Model")
    st.write("The XSS detection model used in this app is based on deep learning techniques. It has been trained on a diverse dataset of sentences to classify them as either XSS attacks or benign. The model takes into account various patterns and features to make accurate predictions.")
 

# Prediction page
def prediction():
    st.title("Cross Site Scripting Attack Detection")

    # Get user input
    st.markdown("<h2 style='white-space: nowrap;'>Enter a sentence:</h2>", unsafe_allow_html=True)
    new_sentence = st.text_area("", "<caption onpointerdown=alert(1)>XSS</caption>", height=100)
    st.write("**You Entered**:", new_sentence)
    st.markdown('***Analyzing the entered sentence for potential XSS attacks...***')

    # Preprocess the input and make predictions
    new_data = preprocess_new_data(new_sentence)
    new_predictions = model.predict(new_data)
    binary_prediction = (new_predictions > 0.5).astype(int)

    # Display the result
    if binary_prediction[0][0] == 1:
        st.markdown(
            f"<p style='color:red; font-size:20px;'>Prediction: XSS Attack</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p style='color:green; font-size:20px;'>Prediction: Benign</p>",
            unsafe_allow_html=True
        )

# Streamlit app
def main():
    st.set_page_config(page_title="XSS Detection App", page_icon="🔒")
    # Upload and display image for the home page
    # Sidebar navigation
    pages = {"Home": home, "Prediction": prediction}
    page = st.sidebar.selectbox("Navigate", tuple(pages.keys()))

    # Display the selected page
    pages[page]()

if __name__ == "__main__":
    main()

