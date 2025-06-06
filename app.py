from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TFSMLayer  # Add this import
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.secret_key = 'scorgal'  # Change this to a random secret key

MODEL_PATH = os.path.join('models', 'earth_classifier.keras')

model = None
model_type = None  # Track how the model was loaded

try:
    if os.path.isdir(MODEL_PATH):
        # It's a SavedModel directory, use TFSMLayer
        print(f"Detected SavedModel directory at {MODEL_PATH}, loading with TFSMLayer...")
        model = TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
        model_type = 'tfsm'
        print(f"✅ Model loaded as TFSMLayer from {MODEL_PATH}")
    else:
        # It's a file (.keras or .h5)
        print(f"Detected model file at {MODEL_PATH}, loading with load_model...")
        model = load_model(MODEL_PATH)
        model_type = 'keras'
        print(f"✅ Model loaded with load_model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# Dynamically load class names in the correct order
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/EuroSAT/2750'))
    
    if os.path.exists(DATA_DIR):
        datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)
        train_data = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(224, 224),
            batch_size=1,
            class_mode='categorical',
            subset='training',
            shuffle=False
        )
        CLASS_NAMES = list(train_data.class_indices.keys())
        print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
    else:
        print(f"Data directory not found: {DATA_DIR}")
        # Fallback class names based on your HTML FEATURE_INFO
        CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                      'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
                      'River', 'SeaLake']
except Exception as e:
    print(f"Error loading class names: {e}")
    # Fallback class names
    CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                  'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
                  'River', 'SeaLake']

# Load class indices mapping if available
CLASS_INDICES_PATH = os.path.join('models', 'class_indices.json')
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Sort class names by index to match model output order
    CLASS_NAMES = [None] * len(class_indices)
    for k, v in class_indices.items():
        CLASS_NAMES[v] = k
    print(f"Loaded class names from class_indices.json: {CLASS_NAMES}")
else:
    print(f"class_indices.json not found at {CLASS_INDICES_PATH}")
    # Use fallback class names
    CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                  'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
                  'River', 'SeaLake']

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def alternative_preprocess_image(filepath):
    """Alternative preprocessing without MobileNetV2 preprocessing"""
    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Simple normalization (0-1 range)
        img_array = img_array / 255.0
        
        print(f"Alternative preprocessing - pixel range: [{img_array.min():.2f}, {img_array.max():.2f}]")
        
        return img_array
    except Exception as e:
        print(f"Error in alternative preprocessing: {e}")
        return None

def preprocess_image(filepath):
    """Preprocess image for prediction"""
    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Debug: Print image stats before preprocessing
        print(f"Image shape: {img_array.shape}")
        print(f"Image pixel range before preprocessing: [{img_array.min():.2f}, {img_array.max():.2f}]")
        
        # Apply preprocessing
        img_array = preprocess_input(img_array)
        
        # Debug: Print image stats after preprocessing
        print(f"Image pixel range after preprocessing: [{img_array.min():.2f}, {img_array.max():.2f}]")
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def get_feature_info(class_name):
    """Get additional feature information for better descriptions"""
    # Convert to lowercase for case-insensitive matching
    class_name_lower = class_name.lower()
    
    feature_descriptions = {
        'annualcrop': {
            'description': 'Agricultural land used for crops that are planted and harvested within a year, such as wheat, corn, or soybeans.',
            'features': ['Agriculture', 'Farming', 'Seasonal', 'Food Production']
        },
        'forest': {
            'description': 'Dense woodland areas with significant tree coverage, important for biodiversity and carbon sequestration.',
            'features': ['Nature', 'Trees', 'Wildlife', 'Carbon Sink']
        },
        'herbaceousvegetation': {
            'description': 'Areas dominated by soft-stemmed plants and grasses, including grasslands and meadows.',
            'features': ['Grassland', 'Natural', 'Grazing', 'Biodiversity']
        },
        'highway': {
            'description': 'Major transportation infrastructure including roads, highways and urban transit networks.',
            'features': ['Transportation', 'Infrastructure', 'Urban', 'Traffic']
        },
        'industrial': {
            'description': 'Manufacturing and industrial facilities including factories, warehouses, and processing plants.',
            'features': ['Manufacturing', 'Industry', 'Commercial', 'Development']
        },
        'pasture': {
            'description': 'Grassland areas used for livestock grazing, typically managed for animal agriculture.',
            'features': ['Livestock', 'Grazing', 'Agriculture', 'Rural']
        },
        'permanentcrop': {
            'description': 'Agricultural areas with perennial crops like orchards, vineyards, and tree plantations.',
            'features': ['Orchard', 'Vineyard', 'Perennial', 'Long-term']
        },
        'residential': {
            'description': 'Urban and suburban areas with housing developments and residential neighborhoods.',
            'features': ['Housing', 'Urban', 'Suburban', 'Community']
        },
        'river': {
            'description': 'Water bodies including rivers, streams, lakes, and other freshwater features.',
            'features': ['Water', 'Freshwater', 'Natural', 'Ecosystem']
        },
        'sealake': {
            'description': 'Large water bodies including seas, large lakes, and coastal marine environments.',
            'features': ['Water', 'Marine', 'Coastal', 'Large Water Body']
        }
    }
    
    # Use lowercase keys for matching
    return feature_descriptions.get(class_name_lower, {
        'description': f'Predicted earth feature: {class_name}',
        'features': [class_name]
    })

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

print(f"🔥 model = {model}")
print(f"🔥 CLASS_NAMES = {CLASS_NAMES}")
print(f"🔥 Upload folder exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}")

import traceback  # Add at the top of app.py if not already

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction with enhanced error logging"""
    try:
        print("🔥 /predict route called")

        # Debug: Check if model is loaded
        if model is None:
            print("❌ Model not loaded.")
            return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500

        # Debug: Check file existence
        if 'file' not in request.files:
            print("❌ No file in request.")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            print("❌ Empty file name.")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF.'}), 400

        # Generate secure filename and path
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        print(f"📁 Saving file to: {filepath}")
        file.save(filepath)

        try:
            print("📷 Starting image preprocessing...")
            img_array = preprocess_image(filepath)
            if img_array is None:
                print("❌ Primary preprocessing failed.")
                return jsonify({'error': 'Failed to process image. Please try another image.'}), 400

            print("🤖 Running model prediction...")

            # Use correct prediction method depending on model type
            if model_type == 'tfsm':
                # TFSMLayer expects a dict input with key 'inputs'
                preds = model({'inputs': img_array}).numpy()
            else:
                preds = model.predict(img_array, verbose=0)

            max_prob = np.max(preds[0])
            if max_prob > 0.99:
                print("⚠️ Suspiciously high confidence — trying alternative preprocessing")
                img_array_alt = alternative_preprocess_image(filepath)
                if img_array_alt is not None:
                    preds_alt = model.predict(img_array_alt, verbose=0)
                    if np.max(preds_alt[0]) < max_prob:
                        preds = preds_alt
                        print("✅ Used alternative predictions")

            print(f"📊 Raw predictions: {preds[0]}")
            print(f"🔢 Prediction shape: {preds.shape}")

            if np.all(np.isnan(preds[0])) or np.all(preds[0] == 0):
                print("❌ Model returned invalid predictions")
                return jsonify({'error': 'Model produced invalid predictions. Please check the model.'}), 500

            pred_idx = np.argmax(preds[0])
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[0][pred_idx]) * 100

            print(f"✅ Predicted: {pred_class} ({confidence:.2f}%)")

            top_indices = np.argsort(preds[0])[-5:][::-1]
            top_predictions = [{
                'label': CLASS_NAMES[idx],
                'confidence': float(preds[0][idx]) * 100
            } for idx in top_indices]

            for i, item in enumerate(top_predictions):
                print(f"Top {i+1}: {item['label']} ({item['confidence']:.2f}%)")

            feature_info = get_feature_info(pred_class)

            response_data = {
                'prediction': pred_class,
                'confidence': confidence,
                'description': feature_info['description'],
                'features': feature_info['features'],
                'top_predictions': top_predictions
            }

            return jsonify(response_data)

        except Exception as e:
            print("🔥 PREDICTION INNER ERROR:")
            traceback.print_exc()
            return jsonify({'error': f'Failed to analyze image: {str(e)}'}), 500

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"🧹 Removed uploaded file: {filepath}")

    except Exception as e:
        print("🔥 GENERAL PREDICT ROUTE ERROR:")
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Earth Feature Classifier Flask App...")
    print(f"Model loaded: {'Yes' if model is not None else 'No'}")
    print(f"Available classes: {len(CLASS_NAMES)}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)