#!/usr/bin/env python3
"""
Test script to diagnose model loading issues
"""

import os
import sys
import traceback
import tensorflow as tf
import keras
import numpy as np

def test_model_loading():
    """Test different methods of loading the model"""
    
    print("=== Environment Info ===")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Check file structure
    print("=== File Structure ===")
    if os.path.exists('models'):
        print("Models directory exists")
        files = os.listdir('models')
        print(f"Files in models/: {files}")
        
        for file in files:
            filepath = os.path.join('models', file)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  {file}: {size} bytes")
    else:
        print("Models directory does not exist!")
        return
    print()
    
    # Test different model files
    model_paths = [
        'models/earth_classifier.keras',
        'models/earth_classifier.h5',
        'models/earth_classifier'
    ]
    
    for model_path in model_paths:
        print(f"=== Testing {model_path} ===")
        
        if not os.path.exists(model_path):
            print(f"❌ {model_path} does not exist")
            continue
            
        print(f"✅ {model_path} exists")
        
        # Test different loading methods
        loading_methods = [
            ("Standard load", lambda: keras.models.load_model(model_path)),
            ("Load without compile", lambda: keras.models.load_model(model_path, compile=False)),
            ("Load with safe_mode=False", lambda: keras.models.load_model(model_path, compile=False, safe_mode=False)),
        ]
        
        if model_path.endswith('.keras'):
            loading_methods = loading_methods  # All methods apply
        elif model_path.endswith('.h5'):
            loading_methods = loading_methods[:2]  # Skip safe_mode for .h5
        elif os.path.isdir(model_path):
            loading_methods = [("SavedModel load", lambda: tf.saved_model.load(model_path))]
        
        for method_name, load_func in loading_methods:
            try:
                print(f"  Trying {method_name}...")
                model = load_func()
                print(f"  ✅ {method_name} succeeded!")
                
                # Test basic model info
                if hasattr(model, 'summary'):
                    print(f"  Model type: {type(model)}")
                    print(f"  Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'Unknown'}")
                    print(f"  Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'Unknown'}")
                    
                    # Test prediction with dummy data
                    try:
                        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                        prediction = model(dummy_input, training=False)
                        print(f"  ✅ Test prediction succeeded! Output shape: {prediction.shape}")
                        return model  # Return the first working model
                    except Exception as pred_error:
                        print(f"  ⚠️  Prediction test failed: {pred_error}")
                elif hasattr(model, 'signatures'):
                    print(f"  SavedModel signatures: {list(model.signatures.keys())}")
                    if 'serving_default' in model.signatures:
                        serving_fn = model.signatures['serving_default']
                        print(f"  Input signature: {serving_fn.structured_input_signature}")
                        return model
                
                break  # If one method works, move to next model
                
            except Exception as e:
                print(f"  ❌ {method_name} failed: {e}")
                if "deserialized properly" in str(e):
                    print(f"  🔍 This is the deserialization error we're debugging!")
        print()
    
    print("❌ All model loading attempts failed!")
    return None

def test_alternative_approaches():
    """Test alternative approaches if standard loading fails"""
    
    print("=== Alternative Approaches ===")
    
    # Try to examine the model file structure
    keras_file = 'models/earth_classifier.keras'
    if os.path.exists(keras_file):
        try:
            import zipfile
            print(f"Examining {keras_file} structure...")
            with zipfile.ZipFile(keras_file, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"Files in .keras archive: {file_list}")
                
                # Check if config.json exists
                if 'config.json' in file_list:
                    config_content = zip_ref.read('config.json').decode('utf-8')
                    print("Model config found:")
                    print(config_content[:500] + "..." if len(config_content) > 500 else config_content)
                    
        except Exception as e:
            print(f"Failed to examine .keras file: {e}")
    
    # Try creating a simple test model
    print("\nTrying to create a simple test model...")
    try:
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(10, activation='softmax')(x)  # 10 classes as per your CLASS_NAMES
        
        test_model = Model(inputs=base_model.input, outputs=predictions)
        print("✅ Test model creation succeeded!")
        
        # Test prediction
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = test_model(dummy_input, training=False)
        print(f"✅ Test prediction succeeded! Output shape: {prediction.shape}")
        
        return test_model
        
    except Exception as e:
        print(f"❌ Test model creation failed: {e}")
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    print("Starting model loading diagnostics...\n")
    
    # Test loading existing models
    model = test_model_loading()
    
    # If that fails, try alternatives
    if model is None:
        model = test_alternative_approaches()
    
    if model is not None:
        print("\n🎉 Success! A working model was found.")
    else:
        print("\n😞 No working model could be loaded or created.")
        print("\nRecommendations:")
        print("1. Check if your model file is corrupted")
        print("2. Try re-saving your model in a different format")
        print("3. Check TensorFlow/Keras version compatibility")
        print("4. Consider using the fallback model approach")