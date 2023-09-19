import os
import gdown
import zipfile
import shutil

os.makedirs('data', exist_ok=True)
print('Downloading scences...')
data_url = 'https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG'
gdown.download(data_url, output='data/nerf_synthetic.zip', quiet=False)
print('Extracting scenes...')
os.makedirs('data/', exist_ok=True)
with zipfile.ZipFile('data/nerf_synthetic.zip', 'r') as zip_file:
    zip_file.extractall('data/')

shutil.rmtree('data/__MACOSX')
shutil.rmtree('data/nerf_synthetic/hotdog')
shutil.rmtree('data/nerf_synthetic/materials')
os.remove('data/nerf_synthetic.zip')
print('Done.')