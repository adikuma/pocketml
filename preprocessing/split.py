import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    # convert inputs to numpy arrays for indexing
    X_arr = np.array(X)
    y_arr = np.array(y)

    if shuffle:
        rng = np.random.RandomState(random_state)
        idx = np.arange(len(X_arr))
        rng.shuffle(idx)
        X_arr = X_arr[idx]
        y_arr = y_arr[idx]

    # split into train and test
    split = int(len(X_arr) * (1 - test_size))
    X_train = X_arr[:split]
    X_test  = X_arr[split:]
    y_train = y_arr[:split]
    y_test  = y_arr[split:]

    return X_train, X_test, y_train, y_test