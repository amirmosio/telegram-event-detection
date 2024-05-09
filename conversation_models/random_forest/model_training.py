import random
import matplotlib
import matplotlib.pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from treeinterpreter import treeinterpreter
import waterfall_chart


def train_random_forest_model(X_train, y_train):
    print(f"Training on {len(y_train)} data point")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

    return clf


def print_model_evaluation(clf, X, y):
    y_pred = clf.predict(X)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Confusion matrix:", confusion_matrix(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("AUC:", roc_auc_score(y, y_pred))

    """
    Confusion matrix graph, for charts
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    """


def print_evaluation_on_test(clf, x, y):
    print_model_evaluation(clf, x, y)

    import warnings

    warnings.simplefilter("ignore", FutureWarning)

    for i in random.sample(range(0, len(x)), 5):
        row = x.iloc[i : i + 1]
        label = y.iloc[i : i + 1].array[0]
        predicted_label = clf.predict(row)[0]
        prediction, bias, contributions = treeinterpreter.predict(clf, row)
        prediction[0], bias[0]

        contributions = [contributions[0][i][1] for i in range(len(contributions[0]))]

        my_plot = waterfall_chart.plot(
            ["bias"] + list(x.columns.array),
            [bias[0][1] - 0.5] + contributions,
            rotation_value=90,
            # threshold=0.3,
            formatting="{:,.3f}",
            net_label="End Result",
            other_label="Remaining Vars",
            Title=f"How does each variable effect the outcome?\n true label {label} predicted {predicted_label}",
            x_lab="Variables",
            y_lab="Prediction",
            green_color="#247747",
            blue_color="#0e4f66",
            red_color="#ff0000",
        )
    matplotlib.pyplot.show()
