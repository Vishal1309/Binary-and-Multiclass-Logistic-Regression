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

for fit_intercept in [True]:
    for lrtype in ['constant','inverse']:
        for batchsize in [30]:
            for lmbda in [0, 0.5]:
                LR = binary_logistic_regression(
                    fit_intercept=fit_intercept, lmbda=lmbda)
                # here you can use fit_non_vectorised / fit_autograd methods
                LR.logistic_regression(
                    X_train1, y_train1, batchsize, lr_type=lrtype)
                y_pred = LR.predict_class(X_test1)
                y_pred_train = LR.predict_class(X_train1)
                # print(y_pred)
                acc_test = sklearn.metrics.accuracy_score(
                    y_test1, y_pred) * 100
                acc_train = sklearn.metrics.accuracy_score(
                    y_train1, y_pred_train) * 100
                print('Fit intercept=', fit_intercept, ', lr_type=', lrtype, ', Batch size=',
                      batchsize, 'lambda=', lmbda, 'Train Accuracy: ', acc_train)
                print('Fit intercept=', fit_intercept, ', lr_type=', lrtype,
                      ', Batch size=', batchsize, 'lambda=', lmbda, 'Test Accuracy: ', acc_test)
                # print('Fit intercept=',fit_intercept,', lr_type=',lrtype,', Batch size=',batchsize,', MAE: ', mae)
                LR.plot_loss()
                LR.decision_surface(X_train,y_train)
