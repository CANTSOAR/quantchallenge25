from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


def sklearn_model(data_csv, model_id = 0):
    """
    model_id: 0 = linear regression, 1 = lasso, 2 = ridge, 3 = xgboost
    """
    data = pd.read_csv(data_csv)
    model = [LinearRegression(), Lasso(), Ridge(), MultiOutputRegressor(GradientBoostingRegressor(n_estimators = 500))][model_id]

    y = data[["Y1", "Y2"]]
    X = data.drop(columns = ["Y1", "Y2"])

    model = model.fit(X, y)

    return model, model.score(X, y)