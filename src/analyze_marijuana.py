import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans


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

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)

    return my_format

if __name__ == '__main__':
    data = pd.read_csv("../data/Marijuana_Arrests.csv")
    labels = 'Avant loi 2015', 'Après loi 2015'

    fig, ax = plt.subplots()
    print("Dataset:\n", type(data))
    sub_data = data.loc[:, ['YEAR']].values
    sub_data = pd.DataFrame(sub_data)
    sizes = [sub_data[sub_data < 2015].count().values[0], sub_data[sub_data > 2014].count().values[0]]
    # median, mean = dataset.median(), dataset.mean()
    median = [sub_data[sub_data < 2015].median().values[0], sub_data[sub_data > 2014].median().values[0]]
    mean = [sub_data[sub_data < 2015].mean().values[0], sub_data[sub_data > 2014].mean().values[0]]
    print(median)
    print(mean)
    ax.pie(sizes, labels=labels, autopct=autopct_format(sizes))
    plt.show()
    """
    # define dataset
    data = pd.read_csv("../data/Marijuana_Arrests.csv")
    data = data.loc[:, ['YEAR']]
    X = data.values
    #X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                               #random_state=4)
    # define the model
    model = KMeans(n_clusters=2)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(x=X[row_ix, 0], y=X[row_ix, 1])
    # show the plot
    plt.show()

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
