import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model("C:/Users/Alex/Desktop/coding/CatsDogsAI/model.h5")

# Prepare an image
image = Image.open("C:/Users/Alex/Desktop/coding/CatsDogsAI/example.jpg")
image = image.resize((224, 224))  # Resize the image to match the input shape of the model
image_array = np.array(image) / 255.0  # Convert the image to a numpy array and normalize the pixel values

# Add an extra dimension to the image array to match the input shape of the model
image_array = np.expand_dims(image_array, axis=0)

# Make a prediction with the model
prediction = model.predict(image_array)

# Convert the prediction to binary class
binary_prediction = (prediction > 0.5).astype("int")

# Map the binary prediction to class labels
class_labels = {0: "cat", 1: "dog"}
predicted_label = class_labels[binary_prediction[0][0]]

print("The image is predicted to be a", predicted_label)
