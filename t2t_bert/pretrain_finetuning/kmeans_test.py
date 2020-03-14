import numpy as np
def kmeans(X, clusters, max_iter):

	random_idx = np.random.permutation(X.shape[0])
	centroids = X[random_idx[:clusters]]

	def get_centers(X, labels):
		centroids = np.zeros((clusters, X.shape[1]))
		for k in range(clusters):
			centroids[k, :] = np.mean(X[labels == k, :], axis=0)
		return centroids
		
	def get_distance(X, centroids):
		distance = np.zeros((X.shape[0], clusters))
		for k in range(clusters):
			row_norm = norm(X - centroids[k, :], axis=1)
			distance[:, k] = np.square(row_norm)

		return distance

	def get_closest_cluster(distance):
		cluster = np.argmin(distance, axis=1)
		return cluster

	def sse(X, labels, centroids):
		distance = np.zeros(X.shape[0])
        for k in range(centroids):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return distance

    def em(centroids, max_iter):
        for i in range(max_iter):
            old_centroids = centroids
            distance = get_distance(X, old_centroids)
            labels = get_closest_cluster(distance)
            centroids = get_centers(X, labels)
            if np.all(old_centroids == centroids):
                break
       error = sse(X, labels, centroids)

       return error, centroids

    error, centroids = em(centroids, max_iter)
    return error, centroids
    

