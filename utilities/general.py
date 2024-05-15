from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def split_train_test_validation(X, y, test_ratio=0.15, val_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def one_hot_encoding(labels):
    label_encoder = LabelEncoder()
    labels_int = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    one_hot_encoded = np.eye(num_classes)[labels_int]
    
    
    unique_labels, label_counts = np.unique(labels_int, return_counts=True)    
    for label in unique_labels:
        original_label = label_encoder.inverse_transform([label])[0]
        print(f"Label: {label}, Original Label: {original_label}, Count: {label_counts[label]}")

    return one_hot_encoded

def fit_transform(labels):
    label_encoder = LabelEncoder()
    labels_int = label_encoder.fit_transform(labels)
    return labels_int