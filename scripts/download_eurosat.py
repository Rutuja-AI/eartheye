# EuroSAT Dataset Download Script
# This script downloads and extracts the EuroSAT RGB dataset into the data/ folder.

import os
import urllib.request
import zipfile

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

dataset_url = 'http://madm.dfki.de/files/sentinel/EuroSAT.zip'
zip_path = os.path.join(data_dir, 'EuroSAT.zip')
extract_path = os.path.join(data_dir, 'EuroSAT')

print('Downloading EuroSAT dataset...')
urllib.request.urlretrieve(dataset_url, zip_path)
print('Download complete!')

print('Extracting dataset...')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print('Extraction complete!')

os.remove(zip_path)
print('Zip file removed.')

print('EuroSAT dataset is ready in the data/EuroSAT directory.')
