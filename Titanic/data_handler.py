import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def remove_incomplete_data(X):
    """
    Remove:
        Cabin - Too many NA
    Age - Could maybe extrapolate from family name and sibsp/parch
    """

    # Fix up null values
    X = X.drop(['Cabin', 'Ticket'], axis=1)
    # Update age with random values selected from age
    age = X['Age']
    val = np.ravel(age.values)
    val = val[~np.isnan(val)]
    X['Age'] = X['Age'].apply(lambda l: l if not pd.isnull(l) else np.random.choice(val))
    # Set embarked null values to most frequent
    X['Embarked'] = X['Embarked'].apply(lambda l: l if not pd.isnull(l) else X['Embarked'].mode()[0])
    # Set fare null to mean
    X['Fare'] = X['Fare'].apply(lambda l: l if not pd.isnull(l) else X['Fare'].mean())

    # Encode non numeric values
    X = X.replace({
        "Sex": {"male": 0, "female": 1},
        "Embarked": {'S': 0, 'C': 1, 'Q': 2}
    })
    name_lst = [
        lambda ms: 1 if type(ms) is str and "Mrs." in ms else ms,
        lambda mr: 2 if type(mr) is str and any(x in mr for x in ["Mr.", "Mme."]) else mr,
        lambda mss: 3 if type(mss) is str and any(x in mss for x in ["Mrs.", "Mlle.", "Ms."]) else mss,
        lambda mst: 4 if type(mst) is str and "Master." in mst else mst,
        lambda l: 0 if type(l) == str else l
    ]
    tmp_name = X['Name']
    for i in name_lst:
        tmp_name = tmp_name.apply(i)
    X['Name'] = tmp_name

    return X


def read_data():
    X = pd.read_csv("data/train.csv", index_col=False)
    X_test = pd.read_csv("data/test.csv", index_col=False)
    print(len(X_test))
    X = remove_incomplete_data(X)
    X_test = remove_incomplete_data(X_test)
    return X, X_test


if __name__ == '__main__':
    read_data()