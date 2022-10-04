import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

import jax.numpy as jnp
import jax

class multiclass_logistic_regression():
    def __init__(self,fit_intercept=True,lmbda = 0):
        self.fit_intercept = fit_intercept 
        self.lmbda = lmbda
        # self.all_coef=[]
        self.loss = []
        self.coef = None

    def softmax(self,x,theta_class,thetas):
        denominator = 0
        for theta_ in thetas:
            denominator += jnp.exp(jnp.dot(x,theta_))
        return (jnp.exp(jnp.dot(x,theta_class))/(denominator))
    
    def multiclass_crossentropy_loss(self,thetas,x,y,k):
        # loss = 0
        # for theta in thetas:
        #   y_pred = self.softmax(x.values,theta,thetas)
        #   y_pred = jnp.array(y_pred)
        #   y = jnp.array(y)
        #   loss += jnp.dot(y.T,jnp.log(y_pred))
        loss = 0
        for i, yi in enumerate(y):
            for cls in range(1, k + 1):
                if(yi == cls):
                    loss -= self.softmax(x.values, thetas[cls], thetas)

        
        loss = loss + self.lmbda * jnp.sum(jnp.dot(thetas.T,thetas))
        return loss


    def multiclass_regression(self, X, y, batch_size, k,n_iter=200, lr=0.01, lr_type='constant'):

        # self.iter = n_iter
        X_copy=X.copy()
        if (self.fit_intercept==True):
            X_copy.insert(0,-1,np.ones((X_copy.shape[0],)))
            X_copy.columns = np.arange(X_copy.shape[1])
        np.random.seed(42)
        self.coef_=np.random.normal(0, 1, size=X_copy.shape[1])
        thetas=self.coef_

        index=0 
        loss_grads = jax.grad(self.crossentropy_loss)             #computes gradients
        self.loss.append(self.crossentropy_loss(thetas, X_copy, y))

        for i in range(n_iter):
            if (index>=X_copy.shape[0]):
                index=0

            if (lr_type=='inverse'):
                lr=lr/(i+1)

            X_1=X_copy.iloc[index:index+batch_size,:]
            y_1=y[index:index+batch_size]

            thetas = thetas - (lr*loss_grads(thetas, X_1, y_1))    #updates thetas using gradients calculated by inbuilt function
            # self.all_coef.append(thetas)
            # print(thetas)
            self.loss.append(self.crossentropy_loss(thetas, X_copy, y))
            index = index + batch_size 

            self.coef_ = thetas

        pass


    def predict_prob(self,X):
        X_copy=X.copy()
        if(len(self.coef_)!=X_copy.shape[1]):      #adding column of 1 for bias
            X_copy.insert(0,-1,np.ones((X_copy.shape[0],)))
            X_copy.columns = np.arange(X_copy.shape[1])
        return self.sigmoid(X_copy.values,self.coef_)
    
    def predict_class(self,y):
        y_pred = np.array(self.predict_prob(y))
        y_pred[y_pred>=0.5] = True
        y_pred[y_pred<0.5] = False
        
        return y_pred

    def plot_loss(self):
        print(self.loss)
        fig = plt.figure(figsize=(10,8))
        plt.plot(self.loss,label=self.lmbda)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title('LossVsIterations')
        plt.legend()
        # plt.show()
        plt.savefig('Question1_LossVsIterations_lambda' + str(self.lmbda)+'.png')

    def decision_surface(self, X, y):
        xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
        Z = self.predict_prob(pd.DataFrame(np.vstack((xx.ravel(), yy.ravel())).T))
        Z = Z.reshape(xx.shape)
        fig,ax = plt.subplots(1,1)
        pl = ax.contourf(xx, yy, Z, levels = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]),linewidths=1, cmap='coolwarm')
        fig.colorbar(pl)
        ax.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap='coolwarm', edgecolors=(0, 0, 0))
        fig.show()
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Decision Surface')
        fig.savefig('Question1_DecisionSurface_lambda' + str(self.lmbda)+'.png')
