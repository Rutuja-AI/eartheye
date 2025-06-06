# test_keras3_model.py - Test the Keras 3 compatible loading

import tensorflow as tf
import keras
import os
import numpy as np
import json

def test_all_formats():
    """Test loading model in different formats for Keras 3"""
    
    print(f"🔍 TensorFlow version: {tf.__version__}")
    print(f"🔍 Keras version: {keras.__version__}")
    
    formats_to_test = [
        ('earth_classifier.keras', 'Keras 3 format'),
        ('earth_classifier.h5', 'HDF5 format'),
        ('earth_classifier', 'SavedModel with TFSMLayer')
    ]
    
    for filename, description in formats_to_test:
        model_path = os.path.join('models', filename)
        print(f"\n🧪 Testing {description}: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ File/folder not found")
            continue
            
        try:
            if filename.endswith('.keras') or filename.endswith('.h5'):
                # Standard loading for .keras and .h5
                model = keras.models.load_model(model_path)
            else:
                # TFSMLayer for SavedModel
                inputs = keras.Input(shape=(224, 224, 3))
                tfsm_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                outputs = tfsm_layer(inputs)
                model = keras.Model(inputs, outputs)
            
            print(f"✅ Loaded successfully!")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            
            # Test prediction
            dummy_input = np.random.random((1, 224, 224, 3))
            predictions = model.predict(dummy_input, verbose=0)
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Prediction sum: {np.sum(predictions):.4f}")
            
            return model  # Return the first working model
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n😞 No model format worked with Keras 3")
    return None

if __name__ == "__main__":
    model = test_all_formats()
    if model:
        print(f"\n🎉 Found working model! You can use this format in your app.")