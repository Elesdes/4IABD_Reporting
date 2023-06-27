import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def calculate_cost(X, centroids, cluster):
    sum = 0
    for i, val in enumerate(X):
        sum += np.sqrt((centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)
    return sum


def kmeans(X, k):
    diff = 1
    cluster = np.zeros(X.shape[0])
    centroids = data.sample(n=k).values
    while diff:
        # for each observation
        for i, row in enumerate(X):
            mn_dist = float('inf')
            # dist of the point from all centroids
            for idx, centroid in enumerate(centroids):
                d = np.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
                # store closest centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
        # if centroids are same then leave
        if np.count_nonzero(centroids - new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster


if __name__ == '__main__':



    """
    data = pd.read_csv("../data/Marijuana_Arrests.csv")
    data = data.loc[:, ['OFFENSE_BLOCKX', 'OFFENSE_BLOCKY']]
    X = data.values
    sns.scatterplot(x=X[:, 0], y=X[:, 1])
    plt.xlabel('OFFENSE_BLOCKX')
    plt.ylabel('OFFENSE_BLOCKY')
    plt.show()

    cost_list = []
    for k in range(1, 10):
        centroids, cluster = kmeans(X, k)
        # WCSS (Within cluster sum of square)
        cost = calculate_cost(X, centroids, cluster)
        cost_list.append(cost)

    print(cost_list)
    sns.lineplot(x=range(1, 10), y=cost_list, marker='o')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()

    k = 4
    centroids, cluster = kmeans(X, k)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cluster)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], s=100, color='y')
    plt.xlabel('Income')
    plt.ylabel('Loan')
    plt.show()
    """
