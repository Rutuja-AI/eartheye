import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/EuroSAT/2750'))
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/earth_classifier.keras'))

# Parameters
img_size = (224, 224)  # Match MobileNetV2 input size
batch_size = 32

# Load validation data
datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)
val_data = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load model
model = load_model(model_path)

# Predict
pred_probs = model.predict(val_data)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Print classification report
print(classification_report(true_classes, pred_classes, target_names=class_labels))
