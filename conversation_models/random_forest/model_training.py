from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_random_forest_model(X_train, y_train):
    print(f"Training on {len(y_train)} data point")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

    return clf


def print_model_evaluation(clf, X, y):
    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    sens = recall_score(y, y_pred)
    prec = precision_score(y, y_pred)

    print("Accuracy:", acc)
    print("Confusion matrix:", cm)
    print("Sensitivity:", sens)
    print("Precision:", prec)

    '''
    Confusion matrix graph, for charts
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    '''