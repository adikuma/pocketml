import numpy as np

class GaussianAnomalyDetector:
    def __init__(self, percentile=1):
        # percentile for epsilon selection on training data
        self.percentile = percentile
        self.mu = None
        self.var = None
        self.epsilon = None

    def fit(self, X_normal):
        self.mu  = X_normal.mean(axis=0)
        self.var = X_normal.var(axis=0)
        # compute likelihoods on training normals
        p = self._likelihood(X_normal)
        # select epsilon at given percentile
        self.epsilon = np.percentile(p, self.percentile)
        return self

    def _likelihood(self, X):
        coeff    = 1.0 / np.sqrt(2*np.pi*self.var)
        exponent = np.exp(- (X - self.mu)**2 / (2*self.var))
        return np.prod(coeff * exponent, axis=1)

    def predict(self, X):
        p = self._likelihood(X)
        return (p < self.epsilon).astype(int)