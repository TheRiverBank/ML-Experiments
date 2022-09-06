import numpy as np
import pandas as pd


def remove_incomplete_data(X):
    """
    Remove:
        Cabin - Too many NA
    Age - Could maybe extrapolate from family name and sibsp/parch
    """
    #print(X[X.isnull().any(axis=1)])  # Check which rows contain NA

    X = X.drop(['Cabin'], axis=1)
    X = X.dropna()

    #print(X[X.isnull().any(axis=1)])  # Confirm no remaining NA

    return X


def read_data():
    X = pd.read_csv("data/train.csv", index_col=False)
    X_test = pd.read_csv("data/test.csv", index_col=False)
    X = remove_incomplete_data(X)


    return X, X_test


if __name__ == '__main__':
    pass