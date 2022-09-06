from sklearn.neighbors import KNeighborsClassifier
from data_handler import read_data


class KNNClassifier():
    def __init__(self, X, y, n_neighbors):
        self.X = X
        self.y = y
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self):
        self.model = self.model.fit(self.X, self.y)

    def classify(self, X_test):
        return self.model.predict(X_test)


if __name__ == '__main__':
   pass