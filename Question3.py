from autogluon.tabular import TabularDataset, TabularPredictor
import jax
import jax.numpy as jnp
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
import numpy as np


# pip install autogluon


rng = np.random.RandomState(0)
X = rng.randn(200, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

print(X.shape, Y.shape)
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y,
            cmap=plt.cm.Paired, edgecolors=(0, 0, 0))

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42)

print(X_train.shape, X_test.shape)


train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1)
train_data.columns = ['x1', 'x2', 'label']
train_data

traindata = TabularDataset(train_data)
traindata = traindata.sample(n=100, random_state=0)
traindata.head()

label = 'label'
print("Summary of class variable: \n", train_data[label].describe())

# Change path
save_path = '/Question3/'
predictor = TabularPredictor(label=label, path=save_path).fit(traindata)

test_data = pd.concat([pd.DataFrame(X_test), pd.Series(y_test)], axis=1)
test_data.columns = ['x1', 'x2', 'label']
testdata = pd.DataFrame(X_test)
testdata.columns = ['x1', 'x2']

test_label = pd.Series(y_test)
# unnecessary, just demonstrates how to load previously-trained predictor from file
predictor = TabularPredictor.load(save_path)

y_pred = predictor.predict(testdata)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(
    y_true=test_label, y_pred=y_pred, auxiliary_metrics=True)

predictor.leaderboard(test_data, silent=True)
