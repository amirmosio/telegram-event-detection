from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def train_random_forest_model(X_train, y_train):
    print(f"Training on {len(y_train)} data point")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

    return clf


def print_model_evaluation(clf, X, y):
    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(acc)
    print(cm)
