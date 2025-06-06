
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import json

# Paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/EuroSAT/2750'))
model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/earth_classifier.keras'))
class_indices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/class_indices.json'))

# Ensure models directory exists
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Parameters
img_size = (224, 224)  # MobileNetV2 expects 224x224
batch_size = 32
epochs = 10

# Load data
datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Transfer learning model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_data, epochs=epochs, validation_data=val_data)

# Save model
model.save(os.path.join(os.path.dirname(model_save_path), 'earth_classifier'), save_format='tf')

print(f"✅ Model saved at: {model_save_path}")

# Save class indices mapping for inference
with open(class_indices_path, 'w') as f:
    json.dump(train_data.class_indices, f, indent=2)
print(f"✅ Class indices saved at: {class_indices_path}")
