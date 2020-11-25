# coding = utf-8

import numpy as np

class KMeans:

    def __init__(self,n_clusters, max_iters=100, random_state=666):
        """初始化Kmeans模型"""
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)


    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = np.linalg.norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iters):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)

            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
        return self

    def predict(self, X_predict):
        distance = self.compute_distance(X, old_centroids)
        return self.find_closest_cluster(distance)

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    features, true_labels = make_blobs(
        n_samples=200,
        n_features=2,
        centers=3,
        cluster_std=2.75,
        random_state=42
    )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=3,
        max_iters=300,
        random_state=42
    )
    kmeans.fit(scaled_features)