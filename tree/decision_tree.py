import math
from collections import Counter

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold= threshold
        self.left = left
        self.right = right
        self.value = value

def _class_prob(y):
    total = len(y)
    counts = Counter(y)
    return {k: v / total for k, v in counts.items()}

def _entropy(p):
    return sum(-prob * math.log2(prob) for prob in p.values() if prob > 0)

def _information_gain(y_parent, y_left, y_right):
    parent_entropy = _entropy(_class_prob(y_parent))
    left_entropy   = _entropy(_class_prob(y_left))
    right_entropy  = _entropy(_class_prob(y_right))
    n       = len(y_parent)
    n_left  = len(y_left)
    n_right = len(y_right)
    weighted = (n_left / n) * left_entropy + (n_right / n) * right_entropy
    return parent_entropy - weighted

def _get_candidate_splits(col):
    uniq = sorted(set(col))
    if len(uniq) < 2:
        return []
    return [(uniq[i] + uniq[i+1]) / 2 for i in range(len(uniq) - 1)]

def _split_dataset(X, y, feature_index, threshold):
    X_left, y_left, X_right, y_right = [], [], [], []
    for xi, yi in zip(X, y):
        if xi[feature_index] <= threshold:
            X_left.append(xi)
            y_left.append(yi)
        else:
            X_right.append(xi)
            y_right.append(yi)
    return X_left, y_left, X_right, y_right

def _find_best_split(X, y):
    best_gain  = -float("inf")
    best_feat  = None
    best_thresh= None
    # avoid truth-testing `X` directly; use len(X)
    n_features = len(X[0]) if len(X) > 0 else 0

    for feat_idx in range(n_features):
        col = [xi[feat_idx] for xi in X]
        for thresh in _get_candidate_splits(col):
            Xl, yl, Xr, yr = _split_dataset(X, y, feat_idx, thresh)
            if not yl or not yr:
                continue
            gain = _information_gain(y, yl, yr)
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat_idx, thresh

    return best_feat, best_thresh, best_gain

def _build_tree(X, y, max_depth=None, min_samples_split=2, depth=0):
    # if all labels identical → leaf
    if len(set(y)) == 1:
        return DecisionNode(value=y[0])

    # if too deep or too few samples → leaf with majority class
    if (max_depth is not None and depth >= max_depth) or len(y) < min_samples_split:
        majority = Counter(y).most_common(1)[0][0]
        return DecisionNode(value=majority)

    feat, thresh, gain = _find_best_split(X, y)
    if gain is None or gain <= 0:
        majority = Counter(y).most_common(1)[0][0]
        return DecisionNode(value=majority)

    X_left, y_left, X_right, y_right = _split_dataset(X, y, feat, thresh)
    left_child  = _build_tree(X_left, y_left, max_depth, min_samples_split, depth + 1)
    right_child = _build_tree(X_right, y_right, max_depth, min_samples_split, depth + 1)
    return DecisionNode(feature_index=feat, threshold=thresh, left=left_child, right=right_child)

def _predict_sample(node, sample):
    if node.value is not None:
        return node.value
    if sample[node.feature_index] <= node.threshold:
        return _predict_sample(node.left, sample)
    else:
        return _predict_sample(node.right, sample)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = _build_tree(X, y, self.max_depth, self.min_samples_split)

    def predict(self, X):
        return [_predict_sample(self.root, xi) for xi in X]