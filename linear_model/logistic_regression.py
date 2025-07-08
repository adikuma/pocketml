import numpy as np

class LogisticRegression:
    def __init__(self, lr=1e-3, n_iters=10000, tol=1e-6):
        self.lr = lr
        self.n_iters = n_iters
        self.tol = tol
        # parameters
        self.w = None
        self.b = None
        self.cost_history = []

    def _sigmoid(self, z):
        # sigmoid activation
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.n_iters):
            z = X.dot(self.w) + self.b
            h = self._sigmoid(z)
            # gradients
            dw = (1/m) * X.T.dot(h - y)
            db = (1/m) * np.sum(h - y)
            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db
            # cost
            cost = - (1/m) * np.sum(
                y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15)
            )
            self.cost_history.append(cost)
            # check convergence
            if i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                break

    def predict_proba(self, X):
        return self._sigmoid(X.dot(self.w) + self.b)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)