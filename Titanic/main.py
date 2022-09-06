import pandas as pd
from classifier import KNNClassifier
from data_handler import read_data
import numpy as np

def create_result(passenger_ids, predictions):
    print(predictions)
    df = pd.DataFrame(data={"PassengerId": passenger_ids, "Survived": predictions})
    df.to_csv("submission.csv", index=False)


def main():
    X, X_test = read_data()
    X = X.replace({"Sex": {"male": 0, "female": 1}})
    y_train = X["Survived"]
    X_train = X[['Pclass', 'Sex', 'Age']]
    pids = X_test["PassengerId"]
    X_test = X_test[['Pclass', 'Sex', 'Age']]
    X_test = X_test.fillna(20)
    print(X_test[X_test.isnull().any(axis=1)])
    X_test = X_test.replace({"Sex": {"male": 0, "female": 1}})
    KNN = KNNClassifier(X_train, y_train, 4)
    KNN.fit()

    preds = KNN.classify(X_test)

    create_result(pids, preds)


if __name__ == '__main__':
 main()