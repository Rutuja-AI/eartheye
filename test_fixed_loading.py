# test_fixed_loading.py - Test the fixed model loading

import tensorflow as tf
import keras
import os
import numpy as np
import json

def create_keras3_compatible_model():
    """Create a Keras 3 compatible model wrapper"""
    
    # Try .keras format first
    keras_path = os.path.join('models', 'earth_classifier.keras')
    savedmodel_path = os.path.join('models', 'earth_classifier')
    
    # Method 1: Try the converted .keras file
    if os.path.exists(keras_path):
        try:
            print(f"Attempting to load .keras model from: {keras_path}")
            
            # Load the converted model
            loaded_model = keras.models.load_model(keras_path)
            
            # Create a wrapper that handles the output properly
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model
                    self.input_shape = model.input_shape
                    self.output_shape = model.output_shape
                
                def predict(self, inputs, **kwargs):
                    predictions = self.model(inputs, training=False)
                    # Handle different output formats
                    if isinstance(predictions, dict):
                        # If it's a dictionary, get the main output
                        if 'dense_1' in predictions:
                            return predictions['dense_1'].numpy()
                        elif 'output_0' in predictions:
                            return predictions['output_0'].numpy()
                        else:
                            # Get the first value from the dict
                            return list(predictions.values())[0].numpy()
                    else:
                        # Direct tensor output
                        if hasattr(predictions, 'numpy'):
                            return predictions.numpy()
                        else:
                            return predictions
            
            wrapper = ModelWrapper(loaded_model)
            print("✅ Successfully loaded .keras model with wrapper")
            return wrapper
            
        except Exception as e:
            print(f"❌ Failed to load .keras model: {e}")
    
    # Method 2: Use SavedModel with raw TensorFlow
    if os.path.exists(savedmodel_path):
        try:
            print(f"Loading SavedModel directly with TensorFlow from: {savedmodel_path}")
            
            # Load the raw SavedModel
            raw_model = tf.saved_model.load(savedmodel_path)
            serving_fn = raw_model.signatures['serving_default']
            
            # Inspect the function signature
            print("Function signature:")
            print(f"  Inputs: {serving_fn.structured_input_signature[1]}")
            print(f"  Outputs: {serving_fn.structured_outputs}")
            
            # Create wrapper class
            class SavedModelWrapper:
                def __init__(self, serving_fn):
                    self.serving_fn = serving_fn
                    self.input_shape = (None, 224, 224, 3)
                    self.output_shape = (None, 10)
                
                def predict(self, inputs, **kwargs):
                    # Convert numpy to tensor if needed
                    if isinstance(inputs, np.ndarray):
                        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
                    
                    # Get the input key name
                    input_keys = list(self.serving_fn.structured_input_signature[1].keys())
                    input_key = input_keys[0]
                    print(f"Using input key: {input_key}")
                    
                    # Make prediction
                    predictions = self.serving_fn(**{input_key: inputs})
                    print(f"Raw predictions type: {type(predictions)}")
                    print(f"Raw predictions keys: {predictions.keys() if isinstance(predictions, dict) else 'Not a dict'}")
                    
                    # Extract the actual prediction values
                    if isinstance(predictions, dict):
                        # Get the output
                        output_keys = list(predictions.keys())
                        output_key = output_keys[0]
                        print(f"Using output key: {output_key}")
                        result = predictions[output_key]
                    else:
                        result = predictions
                    
                    # Convert to numpy
                    if hasattr(result, 'numpy'):
                        return result.numpy()
                    else:
                        return result
            
            wrapper = SavedModelWrapper(serving_fn)
            print("✅ Successfully loaded SavedModel with wrapper")
            return wrapper
            
        except Exception as e:
            print(f"❌ Failed to load SavedModel: {e}")
    
    print("❌ No compatible model found")
    return None

def test_model():
    """Test the model loading and prediction"""
    print("🧪 Testing fixed model loading...")
    
    model = create_keras3_compatible_model()
    
    if model is None:
        print("❌ No model loaded")
        return False
    
    print(f"✅ Model loaded!")
    print(f"📊 Input shape: {model.input_shape}")
    print(f"📊 Output shape: {model.output_shape}")
    
    # Test with dummy data
    print("\n🧪 Testing prediction...")
    dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    
    try:
        predictions = model.predict(dummy_input)
        print(f"✅ Prediction successful!")
        print(f"📊 Prediction shape: {predictions.shape}")
        print(f"📊 Prediction sum: {np.sum(predictions):.4f}")
        print(f"📊 Max prediction: {np.max(predictions):.4f}")
        print(f"📊 Predicted class: {np.argmax(predictions)}")
        
        # Load class names
        class_indices_path = os.path.join('models', 'class_indices.json')
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            
            # Sort class names by index
            class_names = [None] * len(class_indices)
            for name, idx in class_indices.items():
                class_names[idx] = name
            
            predicted_class = class_names[np.argmax(predictions)]
            print(f"📊 Predicted class name: {predicted_class}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model()