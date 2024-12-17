# utils/data_fetcher.py
import os
import requests
import zipfile
import pandas as pd
import datadotworld as dw

# FUCNTION TO FETCH AND SAVE DATA SET
def fetch_save_dataset(url, file_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        open(file_path, mode='wb').write(response.content)
        print(f'Data fetch and save to {file_path}')
    except requests.exceptions.RequestException as e:
        print(f'Error fetching data from {url}: {e}')

# Extract data downloaded zip file from
# with zipfile.ZipFile(zip1, mode='r') as zip_ref:
#     zip_ref.extract(member=csv_file5, path=data_dir)

# EXTRACT DATA FROM DOWLOADED ZIP FILE
def extract_zip_file(zip_file_path, member_csv_file, extract_to_path):
    try:
        with zipfile.ZipFile(zip_file_path, mode='r') as zip_ref:
            zip_ref.extract(member=member_csv_file, path=extract_to_path)  # Member is name files
        print(f'Extract {member_csv_file} to {extract_to_path}')
    except zipfile.BadZipFile as e:
        print(f'Error extrracting {zip_file_path}: {e}')



# ____________________________________________________________________
# LOAD TO DATA FRAME
def load_dataframe_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        print(f'Dataframe loaded from {file_path}')
        return df
    except Exception as e:
        print(f'Error loading dataframe from {file_path}: {e}')
        return None

# LOAD DATA FRAME FROM URL
def load_dataframe_url(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        print(f'Dataframe loaded from {url}')
        print(df.head())
        return df
    except Exception as e:
        print(f'Error loading dataframe from {url}: {e}')
        return None



