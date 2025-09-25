from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def sklearn_model(data, model_id = 0):
    """
    model_id: 0 = linear regression, 1 = lasso, 2 = ridge, 3 = xgboost
    """
    model = [LinearRegression(),
             Lasso(),
             Ridge(),
             MultiOutputRegressor(GradientBoostingRegressor(n_estimators = 1000, learning_rate = .02, max_depth = 5, validation_fraction = .2, n_iter_no_change = 10, subsample = .8)),
             MultiOutputRegressor(HistGradientBoostingRegressor(max_iter = 1000, learning_rate = .02, max_leaf_nodes = 61, validation_fraction = .2, n_iter_no_change = 10)),``
             MultiOutputRegressor(HistGradientBoostingRegressor(max_iter = 1000, learning_rate = .02, max_leaf_nodes = 61, validation_fraction = .2, l2_regularization = 5, min_samples_leaf = 50, n_iter_no_change = 10))
             ][model_id]

    y = data[["Y1", "Y2"]]
    X = data.drop(columns = ["Y1", "Y2"])

    model = model.fit(X, y)

    return model, model.score(X, y)

def make_submit_file(model, data):

    ids = data[["id"]]
    X = data.drop(columns = ["id"])
    y = pd.DataFrame(model.predict(X), columns = ["Y1", "Y2"])

    data = pd.concat([ids, y], axis = 1)
    n = len(os.listdir("./submissions"))

    data.to_csv(f"./submissions/submit_{n}.csv", index = False)

class Data_Generator(Dataset):
    def __init__(self, X, y, lookback_steps = 5):
        self.X = X
        self.y = y
        self.lookback_steps = lookback_steps

    def __len__(self):
        return len(self.X) - self.lookback_steps
    
    def __getitem__(self, i):
        return torch.tensor(self.X[i: i + self.lookback_steps], dtype=torch.float32), torch.tensor(self.y[i + self.lookback_steps], dtype=torch.float32)

class LSTM(nn.Module):
    def __init__(self, input_dim = 13, output_dim = 2, layers = 1, hidden_dim = 64):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first = True)
        self.dnn = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X, _ = self.lstm(X)
        X = X[:, -1, :]
        return self.dnn(X)
    
    def prep_data(self, data):
        if "Y1" not in data.columns:
            data["Y1"] = np.zeros(len(data))
            data["Y2"] = np.zeros(len(data))

        y = data[["Y1", "Y2"]].values
        X = data.drop(columns = ["Y1", "Y2"]).values

        if not hasattr(self, "process"):
            self.process = Pipeline([
                ("scalar", StandardScaler()),
                ("pca", PCA(n_components = .95))
            ])

            X = self.process.fit_transform(X)
        else:
            X = self.process.transform(X)

        return X, y
    
    def train_model(self, data, epochs = 50, lr = 1e-3, batch_size = 32, lookback_steps = 5):
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = lr)

        self.lookback_steps = lookback_steps
        X, y = self.prep_data(data)
        data = Data_Generator(X, y, lookback_steps)
        loader = DataLoader(data, batch_size = batch_size, shuffle = True)

        for e in range(epochs):
            total_loss = 0
            for X, y in loader:
                optimizer.zero_grad()
                y_pred = self(X)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if e % 1 == 0 or e == epochs - 1:
                print(f"Epoch {e}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    def predict(self, data, batch_size = 32):
        X, y = self.prep_data(data)
        data = Data_Generator(X, y, self.lookback_steps)
        loader = DataLoader(data, batch_size = batch_size, shuffle = False)

        self.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                y = self(X)

        y = torch.cat(preds, dim = 0)

        return y