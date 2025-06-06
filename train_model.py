import os
import json

# Paths - clean and consistent
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
model_save_path = os.path.join(models_dir, 'earth_classifier')  # SavedModel format
class_indices_path = os.path.join(models_dir, 'class_indices.json')

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

# ...existing training code...

# After training, save the model (replace the existing save section):
print("Saving model...")

# Remove any existing model files/folders to avoid conflicts
import shutil
if os.path.exists(model_save_path):
    if os.path.isdir(model_save_path):
        shutil.rmtree(model_save_path)
    else:
        os.remove(model_save_path)

# Save in SavedModel format (recommended)
model.save(model_save_path, save_format='tf')
print(f"✅ Model saved at: {model_save_path}")

# Save class indices mapping for inference
with open(class_indices_path, 'w') as f:
    json.dump(train_data.class_indices, f, indent=2)
print(f"✅ Class indices saved at: {class_indices_path}")

print("\nFinal model directory structure:")
for root, dirs, files in os.walk(models_dir):
    level = root.replace(models_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

# ...rest of your code...
