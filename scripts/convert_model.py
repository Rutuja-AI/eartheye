import keras

# Load your old model (could be .h5, .keras, or SavedModel directory)
model = keras.models.load_model('models/earth_classifier.h5')  # or .keras if that's what you have

# Save as TensorFlow SavedModel format (directory)
model.save('models/earth_classifier_savedmodel', save_format='tf')