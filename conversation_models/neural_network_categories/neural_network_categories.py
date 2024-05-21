from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


from numpy import argmax

from utilities.embeddings import embedding_with_sentence_transformer
from sklearn.preprocessing import LabelEncoder

batch_size = 256
epochs = 1000
patience_early_stop = 80
patience_reduce = 20


def neural_network(X, categories):
    import tensorflow as tf

    X = embedding_with_sentence_transformer(np.array(X))
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, categories, test_size=0.15, random_state=42, stratify=categories
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
    )

    unique, count = np.unique(categories, return_counts=True)
    print("Target labels:", unique)
    for idx, u in enumerate(unique):
        print(f"Class {u} has {count[idx]} samples")

    # change
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(unique))

    y_val = encoder.fit_transform(y_val)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(unique))

    y_test = encoder.fit_transform(y_test)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(unique))

    # change
    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print("Training set shape:\t", X_train.shape, y_train.shape)
    print("Validation set shape:\t", X_val.shape, y_val.shape)
    print("Test set shape:\t\t", X_test.shape, y_test.shape)

    model = train(X_train, y_train, X_val, y_val)
    test(X_test, y_test, model)


# change##############################################
def f11_score(y_true, y_pred):
    from tensorflow.keras import backend as K
    import tensorflow as tf

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # Cast true_positives, predicted_positives, and possible_positives to float32 before the division
    precision = K.cast(true_positives, "float32") / (
        K.cast(predicted_positives, "float32") + K.epsilon()
    )
    recall = K.cast(true_positives, "float32") / (
        K.cast(possible_positives, "float32") + K.epsilon()
    )

    f1_val = tf.where(
        tf.math.is_nan(precision) | tf.math.is_nan(recall),
        0.0,
        2 * (precision * recall) / (precision + recall + K.epsilon()),
    )

    return f1_val


###################################


def build_model(input_shape, output_shape):
    from tensorflow.keras import layers as tfkl
    import tensorflow as tf

    tf.random.set_seed(42)

    # change
    input_layer = tfkl.Input(shape=(input_shape,), name="Input")
    hidden_layer_1 = tfkl.Dense(128, activation="relu", name="Hidden_Layer_1")(
        input_layer
    )
    dropout_1 = tfkl.Dropout(0.3)(hidden_layer_1)
    hidden_layer_2 = tfkl.Dense(64, activation="relu", name="Hidden_Layer_2")(dropout_1)
    dropout_2 = tfkl.Dropout(0.2)(hidden_layer_2)
    hidden_layer_3 = tfkl.Dense(32, activation="relu", name="Hidden_Layer_3")(dropout_2)
    output_layer = tfkl.Dense(output_shape, activation="softmax", name="Output")(
        hidden_layer_3
    )

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[f11_score],
    )

    return model


def train(X_train, y_train, X_val, y_val):
    input_shape = X_train.shape[1]  ##CHECK IF IT IS CORRECT
    output_shape = y_train.shape[1]

    model = build_model(input_shape, output_shape)
    model.summary()

    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks(patience_early_stop, patience_reduce),
    ).history

    plot_result(history)

    return model


def test(X_test, y_test, model):
    test_predictions = model.predict(X_test, verbose=0)
    # change
    y_test = argmax(y_test, axis=-1)
    y_pred = np.argmax(test_predictions, axis=-1)
    print_model_evaluation(y_test, y_pred)


def print_model_evaluation(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    overall_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    overall_recall = recall_score(y_true, y_pred, average="macro")
    overall_f1 = f11_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, zero_division=0)

    print("Confusion Matrix:")
    print(conf_matrix)

    print(f"Overall Precision: {overall_precision}")
    print(f"Overall Recall: {overall_recall}")
    print(f"Overall F1-score: {overall_f1}")

    print("Classification Report:")
    print(class_report)


def plot_result(history):
    plt.figure(figsize=(15, 2))
    plt.plot(history["loss"], label="Training loss", alpha=0.8)
    plt.plot(history["val_loss"], label="Validation loss", alpha=0.8)
    plt.title("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.figure(figsize=(15, 2))
    plt.plot(history["f11_score"], label="Training F1 Score", alpha=0.8)
    plt.plot(history["val_f11_score"], label="Validation F1 Score", alpha=0.8)
    plt.title("F1 Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def callbacks(patience_early_stop, patience_reduce):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience_early_stop,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=patience_reduce,
            factor=0.1,
            min_lr=1e-5,
        ),
    ]

    return callbacks
