# [START OF FILE]
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import os, json, uuid, traceback
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = 'scorgal'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None

def alternative_preprocess_image(filepath):
    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0
    except Exception as e:
        print(f"Alternative preprocessing error: {e}")
        return None

# --- Load Class Names ---
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
               'River', 'SeaLake']
class_indices_path = os.path.join('models', 'class_indices.json')
if os.path.exists(class_indices_path):
    try:
        with open(class_indices_path, 'r') as f:
            indices = json.load(f)
        CLASS_NAMES = [None] * len(indices)
        for k, v in indices.items():
            CLASS_NAMES[v] = k
    except:
        pass

# --- Feature Info Map ---
def get_feature_info(label):
    descs = {
        'annualcrop': ("Agricultural land for seasonal crops.", ['Farming', 'Food']),
        'forest': ("Wooded area with biodiversity.", ['Nature', 'Trees']),
        'herbaceousvegetation': ("Grassland and meadows.", ['Grass', 'Natural']),
        'highway': ("Transportation routes.", ['Urban', 'Road']),
        'industrial': ("Factories and warehouses.", ['Industry']),
        'pasture': ("Grazing areas for livestock.", ['Agriculture']),
        'permanentcrop': ("Orchards, plantations.", ['Perennial', 'Farming']),
        'residential': ("Housing areas.", ['Urban', 'Living']),
        'river': ("Freshwater rivers and streams.", ['Water']),
        'sealake': ("Large water bodies.", ['Marine', 'Ocean'])
    }
    l = label.lower()
    d = descs.get(l, (f"Predicted feature: {label}", [label]))
    return {'description': d[0], 'features': d[1]}

# --- Model Loader ---
def create_model():
    paths = [
        os.path.join('models', 'earth_classifier.keras'),
        os.path.join('models', 'earth_classifier'),
        os.path.join('models', 'earth_classifier.h5'),
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith('.keras'):
                model = keras.models.load_model(path)
                return lambda x: model(x, training=False).numpy()
            elif os.path.isdir(path):
                loaded = tf.saved_model.load(path).signatures['serving_default']
                key = list(loaded.structured_input_signature[1].keys())[0]
                return lambda x: loaded(**{key: tf.convert_to_tensor(x)})[list(loaded(**{key: tf.convert_to_tensor(x)}).keys())[0]].numpy()
        except Exception as e:
            print(f"Model load error: {e}")
            continue
    return None

model = create_model()

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        fname = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(path)

        try:
            img = preprocess_image(path)
            if img is None:
                return jsonify({'error': 'Failed to preprocess image'}), 400
            preds = model(img)
            max_prob = np.max(preds[0])
            if max_prob > 0.99:
                alt = alternative_preprocess_image(path)
                if alt is not None and np.max(model(alt)[0]) < max_prob:
                    preds = model(alt)

            if np.all(np.isnan(preds[0])) or np.all(preds[0] == 0):
                return jsonify({'error': 'Invalid prediction'}), 500

            top5 = np.argsort(preds[0])[-5:][::-1]
            result = {
                'prediction': CLASS_NAMES[int(top5[0])],
                'confidence': float(preds[0][top5[0]]) * 100,
                'top_predictions': [{
                    'label': CLASS_NAMES[i],
                    'confidence': float(preds[0][i]) * 100
                } for i in top5]
            }
            result.update(get_feature_info(result['prediction']))
            return jsonify(result)
        finally:
            if os.path.exists(path):
                os.remove(path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e): return jsonify({'error': 'Max size is 10MB'}), 413

@app.errorhandler(404)
def not_found(e): return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e): return jsonify({'error': 'Server error'}), 500


# Add this debug endpoint to your app.py (temporarily)

@app.route('/debug')
def debug():
    """Debug endpoint to check file structure on server"""
    import os
    
    debug_info = {}
    
    # Check current directory
    debug_info['current_dir'] = os.getcwd()
    debug_info['files_in_root'] = os.listdir('.')
    
    # Check models directory
    models_path = 'models'
    if os.path.exists(models_path):
        debug_info['models_exists'] = True
        debug_info['files_in_models'] = os.listdir(models_path)
        
        # Check specific model files
        keras_path = os.path.join('models', 'earth_classifier.keras')
        savedmodel_path = os.path.join('models', 'earth_classifier')
        
        debug_info['keras_file_exists'] = os.path.exists(keras_path)
        debug_info['savedmodel_dir_exists'] = os.path.exists(savedmodel_path)
        
        if os.path.exists(savedmodel_path):
            debug_info['savedmodel_contents'] = os.listdir(savedmodel_path)
    else:
        debug_info['models_exists'] = False
    
    # Check class indices
    class_indices_path = 'class_indices.json'
    debug_info['class_indices_exists'] = os.path.exists(class_indices_path)
    
    return jsonify(debug_info)
# --- Start App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# [END OF FILE]
