# Run this script once to convert your SavedModel to .keras format
# convert_model.py

import tensorflow as tf
import keras
import os

def convert_savedmodel_to_keras():
    """Convert SavedModel to .keras format for Keras 3 compatibility"""
    
    savedmodel_path = os.path.join('models', 'earth_classifier')
    keras_path = os.path.join('models', 'earth_classifier.keras')
    
    if not os.path.exists(savedmodel_path):
        print(f"❌ SavedModel not found at: {savedmodel_path}")
        return False
    
    try:
        print(f"Loading SavedModel from: {savedmodel_path}")
        
        # Load using TensorFlow (not Keras load_model)
        model = tf.saved_model.load(savedmodel_path)
        
        # Get the inference function
        infer = model.signatures['serving_default']
        
        # Create a new Keras model that wraps the SavedModel
        inputs = keras.Input(shape=(224, 224, 3), name='input_1')
        
        # Create TFSMLayer
        tfsm_layer = keras.layers.TFSMLayer(savedmodel_path, call_endpoint='serving_default')
        outputs = tfsm_layer(inputs)
        
        # Create the Keras model
        keras_model = keras.Model(inputs, outputs)
        
        # Save in .keras format
        print(f"Saving as .keras format to: {keras_path}")
        keras_model.save(keras_path)
        
        print("✅ Successfully converted SavedModel to .keras format!")
        
        # Test the converted model
        print("Testing converted model...")
        test_model = keras.models.load_model(keras_path)
        print(f"✅ Test successful! Model shape: {test_model.input_shape} -> {test_model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    success = convert_savedmodel_to_keras()
    if success:
        print("\n🎉 You can now use the .keras model in your app!")
        print("Update your app.py to load 'models/earth_classifier.keras'")
    else:
        print("\n😞 Conversion failed. You might need to retrain with Keras 3 compatible saving.")