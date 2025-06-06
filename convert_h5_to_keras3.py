import keras

model = keras.models.load_model('earth_classifier.h5')
model.save('earth_classifier.keras', save_format='keras')
