import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time


class ConversationRootModel(torch.nn.Module):
    def __init__(self, input_feature_size=300):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        layers_size = [
            input_feature_size,
            input_feature_size,
            256,
            256,
            128,
            128,
            64,
            64,
            32,
            32,
            16,
            8,
            4,
            2,
            1,
        ]
        for i in range(1, len(layers_size)):
            self.layers.append(torch.nn.Linear(layers_size[i - 1], layers_size[i]))
            if i != len(layers_size) - 1:
                self.layers.append(torch.nn.BatchNorm1d(layers_size[i]))
                self.layers.append(torch.nn.LeakyReLU())
            else:
                self.layers.append(torch.nn.Tanh())

    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return (data + 1) / 2

    def save_model(self, name="model"):
        torch.save(self.state_dict(), f"./trained_models/{name}.pth")

    def load_model(self, name="model"):
        self.load_state_dict(torch.load(f"./trained_models/{name}.pth"))


def _get_data_loader(features_tensor, labels_tensor):

    dataset = TensorDataset(features_tensor, labels_tensor)

    batch_size = 32
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_neural_network_model(X, y, X_v, y_v):
    X = torch.tensor(np.array([x for x in X]), dtype=torch.float)
    y = torch.tensor(np.array([l for l in y]), dtype=torch.int)

    X_v = torch.tensor(np.array([x for x in X_v]), dtype=torch.float)
    y_v = torch.tensor(np.array([l for l in y_v]), dtype=torch.int)

    print(f"Training on {len(y)} data point")
    data_loader = _get_data_loader(X, y)

    learning_rate = 0.0005
    n_epoch = 40

    model = ConversationRootModel(input_feature_size=len(X[0]))
    cec = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_acc_history, val_acc_history = [0.5], [0.5]
    for e in tqdm(range(n_epoch)):
        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            pred = model(inputs)
            loss = cec(pred.flatten(), labels.float())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_acc_history.append(accuracy_score(y, model(X).round()))
            val_acc_history.append(accuracy_score(y_v, model(X_v).round()))
            if max(val_acc_history[:-1]) <= val_acc_history[-1]:
                model.save_model(
                    f"conversation_model_{time.time()}_epoch_{e}_acc_{round(val_acc_history[-1]*100)}"
                )

    return model, train_acc_history, val_acc_history


def draw_loss_chart(train_acc_history, val_acc_history):

    epochs = range(1, len(train_acc_history) + 1)

    plt.plot(epochs, train_acc_history, "r")
    plt.plot(epochs, val_acc_history, "b")
    plt.title("Training Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.show()


def print_model_evaluation(clf, X, y):
    X = torch.tensor(np.array([x for x in X]), dtype=torch.float)
    y = torch.tensor(np.array([l for l in y]), dtype=torch.int)
    with torch.no_grad():
        y_pred = clf(X)

        acc = accuracy_score(y, y_pred.round())
        cm = confusion_matrix(y, y_pred.round())
        sens = recall_score(y, y_pred.round())
        prec = precision_score(y, y_pred.round())

        print("Accuracy:", acc)
        print("Confusion matrix:", cm)
        print("Sensitivity:", sens)
        print("Precision:", prec)
