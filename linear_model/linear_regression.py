import numpy as np

class LinearRegression:
    def __init__(self, lr=1e-3, n_iters=1000, method='batch', shuffle = False):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.method = method
        self.shuffle = shuffle
        self.cost_history = []

    def fit(self, X, y):
        if self.method == "stochastic":
            self._fit_stochastic(X, y)
        elif self.method == "batch":
            self._fit_batch(X, y)
        else:
            raise ValueError("Method must be either 'stochastic' or 'batch'")

    def _fit_batch(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []
        for _ in range(self.n_iters):
            y_pred = self.predict(X)
            dw = (1 / m) * X.T.dot(y_pred - y)
            db = (1 / m) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

    def _fit_stochastic(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []
        
        for _ in range(self.n_iters):
            if self.shuffle:
                indices = np.random.permutation(m)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
                
            for i in range(m):
                xi = X_shuffled[i]
                yi = y_shuffled[i]
                pred_i = xi.dot(self.w) + self.b
                error = pred_i - yi
                dw = xi * error
                db = error
                self.w -= self.lr * dw
                self.b -= self.lr * db

            y_pred = X.dot(self.w) + self.b
            cost = 0.5 * np.mean((y_pred - y)**2)
            self.cost_history.append(cost)
                    
    def predict(self, X):
        return X.dot(self.w) + self.b