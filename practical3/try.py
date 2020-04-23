## from sklearn.cluster import KMeans#we need this, numpy is defined at the top of this file

#class sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10,
# max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')

# def kmeans_generic(d_dim_set, num_clusters):

#     """
#     Implements a generic version of the K-means algorithm

#     @param d_dim_set,d -dimensional data set, i.e. X
#     @param num_clusters-number of clusters into which you wish to group your data set
#     @return k_cluster_means-means of your K-clusters,Initialize your means with two random samples.
#     @return assigned_labels-assigned labels of your dataset (use 1-of-K coding to represent your labels)
#     """

#     kmeans=KMeans(n_clusters=num_clusters,random_state=0).fit(X)
#     assigned_labels=kmeans.labels_
#     kmeans.predict(X)#predict with 2 random points
#     k_cluster_means=kmeans.cluster_centers_

#     return k_cluster_means,assigned_labels

# #KMeans implementation
# k_cluster_means,assigned_labels=kmeans_generic(X, 2)

# print(assigned_labels)
# print(k_cluster_means)

# c2 = np.array(["b","y"])#these are the colours for our kmeans results
# for k in range(k_cluster_means.shape[1]):
#     plt.plot(k_cluster_means[0,k],k_cluster_means[1,k],c2[assigned_labels[k]]+"o")#plot all the points
X=X.T
# plt.title("After kmeans algorithm")
# plt.show()
kmeans=KMeans(n_clusters=2,random_state=0)
y_kmeans=kmeans.fit_predict(X)
print(kmeans)
# Now that we have predicted the cluster labels y_km, let’s visualize the clusters that k-means identified
# in the dataset together with the cluster centroids. These are stored under the cluster_centers_ attribute of the fitted KMeans object:
c2 = np.array(["b","y"])#array of characters we want to split into

for k in range(X.shape[1]):
    plt.plot(X[0,k],X[1,k],c2[y_kmeans[k]]+"o")#'1' is for the marker shape

plt.show()
