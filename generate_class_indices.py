from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

datagen = ImageDataGenerator(validation_split=0.2)
train_data = datagen.flow_from_directory(
    "data/EuroSAT/2750",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

with open("models/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Saved class_indices.json!")
