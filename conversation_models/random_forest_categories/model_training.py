from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utilities.embeddings import embedding_with_sentence_transformer
from utilities.general import (
    one_hot_encoding,
    fit_transform
)
import pandas as pd
import numpy as np

def train_random_forest_model_multiclass(df_train, y_train):
    X_train = embedding_with_sentence_transformer(list(df_train))
    y_train = one_hot_encoding(np.asarray(y_train))
    #print(y_train.value_counts())
    best_parameters, best_model = grid_search_random_forest(X_train, y_train)
    clf = best_model
    '''
    clf = RandomForestClassifier(best_parameters, random_state=42)
    clf.fit(X_train, y_train)
    '''

    return clf

def grid_search_random_forest(X_train, y_train):
    param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20],       
    'min_samples_split': [2, 5, 10],   
    'min_samples_leaf': [1, 2, 4],     
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
    }

    clf_multiclaass = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(estimator=clf_multiclaass, param_grid=param_grid,cv=5, scoring='f1_macro')######

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    best_model = grid_search.best_estimator_
    
    print("Best Parameters:", best_params)
    print("Best Model:", best_model)
    print("Best Score on Training Set - CV with 5 folds:", best_score)

    return best_params, best_model


def print_model_evaluation(clf_multiclaass, df_test, y_true):
    X_test = embedding_with_sentence_transformer(list(df_test))

    y_pred_one_hot = clf_multiclaass.predict(X_test)
    y_pred = np.argmax(y_pred_one_hot,axis=-1)
    y_true = fit_transform(y_true)

    conf_matrix = confusion_matrix(y_true, y_pred)

    overall_precision = precision_score(y_true, y_pred, average='macro')
    overall_recall = recall_score(y_true, y_pred, average='macro')
    overall_f1 = f1_score(y_true, y_pred, average='macro')

    class_report = classification_report(y_true, y_pred)

    print("Confusion Matrix:")
    print(conf_matrix)

    print(f"Overall Precision: {overall_precision}")
    print(f"Overall Recall: {overall_recall}")
    print(f"Overall F1-score: {overall_f1}")

    print("Classification Report:")
    print(class_report)