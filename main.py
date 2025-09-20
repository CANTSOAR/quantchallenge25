from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def sklearn_model(train_data_csv, model_id = 0):
    """
    model_id: 0 = linear regression, 1 = lasso, 2 = ridge, 3 = xgboost
    """
    data = pd.read_csv(train_data_csv)
    model = [LinearRegression(),
             Lasso(),
             Ridge(),
             MultiOutputRegressor(GradientBoostingRegressor(n_estimators = 500, max_depth = 5, validation_fraction = .1, n_iter_no_change = 10))
             ][model_id]

    y = data[["Y1", "Y2"]]
    X = data.drop(columns = ["Y1", "Y2"])

    model = model.fit(X, y)

    return model, model.score(X, y)


def make_submit_file(model, test_data_csv):

    data = pd.read_csv(test_data_csv)

    ids = data[["id"]]
    X = data.drop(columns = ["id"])
    y = pd.DataFrame(model.predict(X), columns = ["Y1", "Y2"])

    data = pd.concat([ids, y], axis = 1)
    n = os.listdir("./submissions")

    data.to_csv(f"./submissions/submit_{n}.csv")
