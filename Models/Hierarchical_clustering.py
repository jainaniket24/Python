# Hierarchical Clustering

# importing the data
import pandas as pd
mall = pd.read_csv('Mall_Customers.csv')
x_var = mall.iloc[:, 3:5].values

# finding the optimal number of clusters using dendogram
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
mall_dendogram = sch.dendrogram(sch.linkage(x_var, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Number of Clusters')
plt.ylabel('Distance')
plt.show()

# we observe that optimal number of  clusters is 5
n_clust = 5

# fitting the hierarchical clustering to the data
from sklearn.cluster import AgglomerativeClustering
mall_clust = AgglomerativeClustering(n_clusters = n_clust, affinity = 'euclidean',
                                     linkage = 'ward')
y_clust = mall_clust.fit_predict(x_var)

# visualizing the clusters
plt_col = {1:'red', 2:'blue', 3:'green', 4:'brown', 5:'black'}
for i in range(n_clust):
    plt.scatter(x_var[y_clust == i, 0], x_var[y_clust == i, 1], s = 50,
            c = plt_col[i+1], label = ('Cluster ' + str(i + 1)))
plt.title('Clusters of Customers in Mall')
plt.legend()
plt.xlabel('Income')
plt.ylabel('Score')
plt.show()