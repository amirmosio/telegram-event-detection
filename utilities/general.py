from sklearn.model_selection import train_test_split


def split_train_test_validation(X, y, test_ratio=0.15, val_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
