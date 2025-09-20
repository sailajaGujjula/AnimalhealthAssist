import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    X = df.drop('Disease', axis=1).values
    y = df['Disease'].values
    return np.column_stack((X, y)), label_encoders
