import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    if shuffle:
        rng = np.random.RandomState(random_state)
        idx = np.arange(len(X))
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]