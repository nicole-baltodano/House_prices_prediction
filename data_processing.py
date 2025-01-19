### data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path, index_col="Id")
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']
    return data, X, y

def split_data(X, y, train_size=0.3, random_state=42):
    return train_test_split(X, y, train_size=train_size, random_state=random_state)
