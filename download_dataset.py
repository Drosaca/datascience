from dotenv import load_dotenv

load_dotenv()

import kaggle

dataset_file_path = 'datasets/training.1600000.processed.noemoticon.csv'


def download_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('kazanova/sentiment140', path='datasets', unzip=True)
    return dataset_file_path

