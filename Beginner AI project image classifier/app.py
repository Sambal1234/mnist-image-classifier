import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
#import cv2 # for image procesing

# 1 configuration
MODEL_PATH = "Beginner AI project image classifier/mnist_model.h5"

# Load the model outside the main function so it only loads once
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop() # stop the app if model can't be loaded

# 2 Pre Processing function
def preprocess_image(image_bytes):
    '''
    converts the uploaded/captured image into 28x28 grayscale,
    inverted format that the MNIST model expects/understands
    '''
    # convert byte to PIL image
    image = Image.open(image_bytes).convert('L') # convert to Grayscale ('L')

    # convert PIL image to Numpy array
    img_array = np.array(image)

    # Invert Colors (The MNIST model expects a white digit on a black background)
    # Webcam: Black ink (~0) on White background (~255)
    # MNIST: White ink (~255) on Black background (~0)
    # So, we simply subtract the array from 255 to invert the brightness.
    inverted_array = 255.0 - img_array

    # Normalize (scale) pixels to values to be between 0 and 1
    normalized_array = inverted_array / 255.0

    # Reshape the array to match the model's expected input shape: (1, 28, 28, 1)
    # (Batch size, height, width, channels)
    final_input = np.expand_dims(normalized_array, axis = -1)
    final_input = np.expand_dims(final_input, axis = 0)

    display_digit = (inverted_array/255.0) *255.0

    return final_input, display_digit

# 3 Streamlit main app layout
st.title("Live webcam MNIST digit classifier")
st.markdown("Use your webam to draw a digit on paper and see the prediction!")

# use the camera input widget
camera_image = st.camera_input("Take a photo of handwritten digit")

if camera_image is not None:
    # a. Show the user the image they took
    st.image(camera_image, caption = "Captured image", use_column_width = True)

    # b. Preprocess the image for the model
    processed_input, display_digit = preprocess_image(camera_image)

    # c. Display the processed image
    st.subheader("Model Input (28x28) Inverted")
    st.image(display_digit, width = 100)

    # d. Predict the digit
    prediction = model.predict(processed_input)
    predicted_class = np.argmax(prediction, axis = 1)[0]
    confidence = np.max(prediction) * 100

    # e. Display the result
    st.success(f"**Predicted Digit: {predicted_class}**")
    st.write(F"Confidence: {confidence: 2f}%")

    # f. Display all probabilities
    st.bar_chart(prediction.flatten())
