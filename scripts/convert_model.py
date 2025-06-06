from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# Load the existing .keras model
keras_model_path = os.path.abspath('../models/earth_classifier.keras')
saved_model_path = os.path.abspath('../models/earth_classifier')  # folder format

model = load_model(keras_model_path)
model.save(saved_model_path, save_format='tf')

print(f"✅ Converted and saved model to SavedModel format at: {saved_model_path}")
