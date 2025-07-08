import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        m, n = X.shape
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        # random init
        self.centroids = np.array([
            np.random.uniform(data_min, data_max)
            for _ in range(self.k)
        ])
        for _ in range(self.max_iters):
            # assign each point to nearest centroid
            dists = np.linalg.norm(
                X[:, None, :] - self.centroids[None, :, :],
                axis=2
            )  # shape (m,k)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels==j)
                else self.centroids[j]
                for j in range(self.k)
            ])
            # check convergence
            if np.allclose(new_centroids, self.centroids, atol=self.tol):
                break
            self.centroids = new_centroids
        self.labels_ = labels
        return self

    def predict(self, X):
        dists = np.linalg.norm(
            X[:, None, :] - self.centroids[None, :, :],
            axis=2
        )
        return np.argmin(dists, axis=1)