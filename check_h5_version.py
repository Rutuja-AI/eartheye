import h5py

filename = 'earth_classifier.h5'  # Change to your file name if needed

with h5py.File(filename, 'r') as f:
    if 'keras_version' in f.attrs:
        print("Keras version:", f.attrs['keras_version'])
    if 'backend' in f.attrs:
        print("Backend:", f.attrs['backend'])
