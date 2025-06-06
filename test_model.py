# test_model.py - Run this to verify your model loads correctly

import tensorflow as tf
import os
import json
import numpy as np

def test_model_loading():
    print("🧪 Testing model loading...")
    
    # Test model loading
    MODEL_PATH = os.path.join('models', 'earth_classifier')
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Check model structure
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        print(f"📊 Number of layers: {len(model.layers)}")
        
        # Test with dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        predictions = model.predict(dummy_input, verbose=0)
        print(f"📊 Prediction shape: {predictions.shape}")
        print(f"📊 Prediction sum: {np.sum(predictions[0]):.4f} (should be ~1.0)")
        
        # Load class indices
        class_indices_path = os.path.join('models', 'class_indices.json')
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            print(f"📊 Number of classes: {len(class_indices)}")
            print(f"📊 Classes: {list(class_indices.keys())}")
        else:
            print("⚠️  class_indices.json not found")
            
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()