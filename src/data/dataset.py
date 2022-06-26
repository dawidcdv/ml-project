import pandas as pd
from src.features.helpers import absolute_path


def load_raw_train_data():
    return pd.read_csv(absolute_path('data', 'raw', 'train_data.csv'),  header=None)


def load_train_data():
    return pd.read_csv(absolute_path('data', 'processed', 'train_data.csv'),  header=None)


def load_labels():
    return pd.read_csv(absolute_path('data', 'raw', 'train_labels.csv'),  header=None)


def load_test_data():
    return pd.read_csv(absolute_path('data', 'raw', 'test_data.csv'),  header=None)
