import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from data_handler import read_data
from seaborn import pairplot
from data_handler import remove_incomplete_data

def plot_classes_for_each_feat(X):
    X_subset = X[['Pclass', 'Sex', 'Age']]

    fig, axs = plt.subplots(1, X_subset.shape[1])
    i = 0

    for feat in X_subset:
        c1 = X[X['Survived'] == 0][feat]
        c2 = X[X['Survived'] == 1][feat]

        axs[i].hist(c1, bins=20, label="Dead")
        axs[i].hist(c2, bins=20, label="Survived")

        i += 1
    plt.legend()
    plt.show()


def pair_plot(X):
    pairplot(X, hue='Survived')
    plt.show()


if __name__ == '__main__':
    X, X_test = read_data()



