from Question2 import model
from tensorflow.keras import regularizers
import tensorflow_addons as tfa


import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from Question1 import binary_logistic_regression
import sklearn

rng = np.random.RandomState(0)
X = rng.randn(200, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# print(X.shape,Y.shape)
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y,
            cmap=plt.cm.Paired, edgecolors=(0, 0, 0))

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42)

# print(X_train.shape,X_test.shape)

X_train1 = pd.DataFrame(X_train)
y_train1 = pd.Series(y_train)
X_test1 = pd.DataFrame(X_test)
y_test1 = pd.Series(y_test)


model.compile(optimizer=tfa.optimizers.AdamW(weight_decay=0.5),
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train1, y_train1, epochs=150, batch_size=30)

y_pred = model.predict(X_test1)
y_pred[y_pred >= 0.5] = True
y_pred[y_pred < 0.5] = False
acc = sklearn.metrics.accuracy_score(y_test1, y_pred)*100
print(acc)
