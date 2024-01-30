import time

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to perform image recognition
def predict(frame):
    # Resize the frame to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = Image.fromarray(frame)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score

# Streamlit app
def main():
    st.title("Zoölozie")

    # Open the webcam (assuming it's the first webcam, change the argument if it's not)
    cap = cv2.VideoCapture(0)

    # Create a placeholder for the webcam frame
    frame_placeholder = st.empty()

    recognition_running = st.checkbox("Start Herkenning")

    # Create a text element to display the prediction
    prediction_text = st.text("")

    # Create a progress bar element
    progress_bar = st.empty()

    while recognition_running:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform image recognition
        class_name, confidence_score = predict(frame)

        # Display the frame in the placeholder
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Update the prediction text
        output_text = f"Klasse: {class_name} \n  ZekerheidScore: {100 * confidence_score:.0f}%"
        prediction_text.text(output_text)

        # Update the progress bar using HTML and CSS
        confidence_percentage = int(confidence_score * 100)
        progress_html = f"""
            <div style="background: {'red' if confidence_score < 0.3 else 'yellow' if confidence_score < 0.7 else 'green'}; width: {confidence_percentage}%; height: 20px;"></div>
        """
        progress_bar.markdown(progress_html, unsafe_allow_html=True)

    time.sleep(0.01666)

    # Release the webcam
    cap.release()

if __name__ == "__main__":
    main()
