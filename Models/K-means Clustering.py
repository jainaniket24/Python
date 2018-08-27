# K-means Clustering

# importing the data set
import pandas as pd
mall = pd.read_csv('Mall_Customers.csv')
x_var = mall.iloc[:, 3:5].values

# using the elbow mwthod to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300,
                    n_init = 10, random_state = 0)
    kmeans.fit(x_var)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Mwthod')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# plot shows that optimal number of clusters is 5
# applying K-means with number of clusters = 5
n_clust = 5
mall_kmeans = KMeans(n_clusters = n_clust, init = 'k-means++', max_iter = 300,
                     n_init = 10, random_state = 0)
y_kmeans = mall_kmeans.fit_predict(x_var)


# visualizig the cluster
plt_col = {1:'red', 2:'blue', 3:'green', 4:'brown', 5:'black'}
for i in range(n_clust):
    plt.scatter(x_var[y_kmeans == i, 0], x_var[y_kmeans == i, 1], s = 50,
            c = plt_col[i+1], label = ('Cluster ' + str(i + 1)))

plt.scatter(mall_kmeans.cluster_centers_[:, 0], mall_kmeans.cluster_centers_[:, 1],
            s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Customers in Mall')
plt.legend()
plt.xlabel('Income')
plt.ylabel('Score')
plt.show()