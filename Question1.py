from genericpath import exists
from venv import create
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf
import os

import jax.numpy as jnp
import jax


class binary_logistic_regression():
    def __init__(self, fit_intercept=True, lmbda=0):
        self.fit_intercept = fit_intercept
        self.lmbda = lmbda
        # self.all_coef=[]
        self.loss = []
        self.coef = None

    def sigmoid(self, x, theta):
        return (1/(1 + jnp.exp(-jnp.dot(x, theta))))

    def crossentropy_loss(self, theta, x, y):
        y_pred = self.sigmoid(x.values, theta)
        y_pred = jnp.array(y_pred)
        y = jnp.array(y)
        loss = -(jnp.dot(y.T, jnp.log(y_pred)) + jnp.dot((jnp.ones(y.shape[0])-y).T, jnp.log(
            jnp.ones(y.shape[0])-y_pred))) + self.lmbda * jnp.dot(theta.T, theta)
        return loss

    def logistic_regression(self, X, y, batch_size, n_iter=150, lr=0.01, lr_type='constant'):

        # self.iter = n_iter
        X_copy = X.copy()
        if (self.fit_intercept == True):
            X_copy.insert(0, -1, np.ones((X_copy.shape[0],)))
            X_copy.columns = np.arange(X_copy.shape[1])
        np.random.seed(42)
        self.coef_ = np.random.normal(0, 1, size=X_copy.shape[1])
        thetas = self.coef_

        index = 0
        loss_grads = jax.grad(self.crossentropy_loss)  # computes gradients
        self.loss.append(self.crossentropy_loss(thetas, X_copy, y))

        for i in range(n_iter):
            if (index >= X_copy.shape[0]):
                index = 0

            if (lr_type == 'inverse'):
                lr = lr/(i+1)

            X_1 = X_copy.iloc[index:index+batch_size, :]
            y_1 = y[index:index+batch_size]

            # updates thetas using gradients calculated by inbuilt function
            thetas = thetas - (lr*loss_grads(thetas, X_1, y_1))
            # self.all_coef.append(thetas)
            # print(thetas)
            self.loss.append(self.crossentropy_loss(thetas, X_copy, y))
            index = index + batch_size

        self.coef_ = thetas

        pass

    def predict_prob(self, X):
        X_copy = X.copy()
        if(len(self.coef_) != X_copy.shape[1]):  # adding column of 1 for bias
            X_copy.insert(0, -1, np.ones((X_copy.shape[0],)))
            X_copy.columns = np.arange(X_copy.shape[1])
            return self.sigmoid(X_copy.values, self.coef_)

    def predict_class(self, y):
        y_pred = np.array(self.predict_prob(y))
        y_pred[y_pred >= 0.5] = True
        y_pred[y_pred < 0.5] = False

        return y_pred

    def plot_loss(self):
        if(os.path.exists('Plots/Question1/') == False):
            os.makedirs('Plots/Question1/')

        print(self.loss)
        fig = plt.figure(figsize=(10, 8))
        plt.plot(self.loss, label=self.lmbda)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title('LossVsIterations')
        plt.legend()
        # plt.show()
        plt.savefig('Plots/Question1/Question1_LossVsIterations_lambda' +
                    str(self.lmbda)+'.png')

    def decision_surface(self, X_train, y_train):
        if(os.path.exists('Plots/Question1/') == False):
            os.makedirs('Plots/Question1/')
        xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
        Z = self.predict_prob(pd.DataFrame(
            np.vstack((xx.ravel(), yy.ravel())).T))
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots(1, 1)
        pl = ax.contourf(xx, yy, Z, levels=np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), linewidths=1, cmap='coolwarm')
        fig.colorbar(pl)
        ax.scatter(X_train[:, 0], X_train[:, 1], s=30,
                   c=y_train, cmap='coolwarm', edgecolors=(0, 0, 0))
        fig.show()
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Decision Surface')
        fig.savefig('Plots/Question1/Question1_DecisionSurface_lambda' +
                    str(self.lmbda)+'.png')
