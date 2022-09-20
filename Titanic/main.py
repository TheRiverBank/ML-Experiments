import pandas as pd
from classifier import KNNClassifier
from data_handler import read_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def create_result(passenger_ids, predictions):
    print(predictions)
    df = pd.DataFrame(data={"PassengerId": passenger_ids, "Survived": predictions})
    df.to_csv("submission.csv", index=False)


def main():
    X, X_test = read_data()
    y_train = X["Survived"]
    X_train = X[['Pclass', 'Sex', 'Age']]
    pids = X_test["PassengerId"]
    X_test = X_test[['Pclass', 'Sex', 'Age']]

    model = RandomForestClassifier(100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    create_result(pids, preds)
    acc_knn = round(model.score(X_train, y_train) * 100, 2)
    print(acc_knn)


if __name__ == '__main__':
 main()