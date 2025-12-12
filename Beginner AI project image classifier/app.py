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
    from PIL import Image
    import numpy as np
    
    # 1. Open, convert to Grayscale ('L'), and resize to 28x28
    image = Image.open(image_bytes).convert('L') 
    image = image.resize((28, 28)) 

    # 2. Convert to Numpy array and Normalize to 0.0-1.0
    img_array = np.array(image)
    normalized_img = img_array.astype('float32') / 255.0

    # 3. Invert Colors (CRITICAL for MNIST accuracy)
    inverted_normalized_img = 1.0 - normalized_img 

    # 🛑 THE FINAL FIX: Prepare Model Input - Reshape to (1, 28, 28)
    # The error says the model expects (None, 28, 28), so we reshape to (1, 28, 28).
    # We do NOT include the channel dimension (the '1' at the end).
    final_input = inverted_normalized_img.reshape(1, 28, 28)
    
    # Ensure the data type is float32
    final_input = final_input.astype('float32')

    # 4. Prepare Display Output
    display_digit = inverted_normalized_img

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
