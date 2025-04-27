import os
import requests
import zipfile
from pathlib import Path

def download_data(url):
    # Set data path
    data_path = Path('data')
    data_path.mkdir(parents=True, exist_ok=True)

    # Define zip file and extraction paths
    zip_path = data_path / 'food_data.zip'
    extract_path = data_path / 'food_data'

    # Check if extracted data already exists
    if extract_path.exists():
        print(f'{extract_path} already exists. Skipping download.')
    else:
        print(f'{extract_path} does not exist. Downloading...')

        # Download data
        response = requests.get(url)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f'Data saved at {zip_path}')

            # Unzip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f'Data unzipped at {extract_path}')

            # Remove zip file
            os.remove(zip_path)
            print('Zip file removed after extraction.')
        else:
            raise Exception('Failed to download data.')

    # Define train and test directories
    train_dir = extract_path / 'train'
    test_dir = extract_path / 'test'

    return extract_path, train_dir, test_dir
