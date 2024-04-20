from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_random_forest_model(df):
    # To make the dataset more balanced and unbiased we have to drop extra label==True records so that
    # number of label==True traning would be equal to number of label=False
    true_df_to_be_dropped = df[df["label"] == True].iloc[sum(df["label"] == False) :]
    new_df = df.drop(true_df_to_be_dropped.index)
    X = new_df.drop("label", axis=1)
    y = new_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred), clf
