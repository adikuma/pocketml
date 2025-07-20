# pocketml

`pocketml` is a simple Python machine learning library implementing basic algorithms in pure Python.

Features:

- Data preprocessing: encoding, scaling, train-test splitting
- Regression: Linear Regression
- Classification: Logistic Regression, Decision Tree Classifier
- Clustering: KMeans
- Anomaly Detection: Gaussian Anomaly Detector

## Installation

Install directly from GitHub:

```
pip install git+https://github.com/<username>/pocketml.git
```

Or clone and install locally:

```
git clone https://github.com/<username>/pocketml.git
cd pocketml
pip install -e .
```

Alternatively, install dependencies and use it locally:

```
pip install -r requirements.txt
pip install -e .
```

## Usage

### Decision Tree Classifier

```python
from pocketml.preprocessing import train_test_split
from pocketml.tree import DecisionTreeClassifier
from pocketml.metrics.classification import accuracy_score

# Sample data
X = [[2.7, 2.5], [1.3, 1.5], [3.1, 2.9], [0.5, 0.7]]
y = [1, 0, 1, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Initialize and train classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Predict and evaluate
predictions = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

For other algorithms and examples, see the `pocketml/usage` directory.

Replace `<username>` with your GitHub username.
