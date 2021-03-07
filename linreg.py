'''
# Name: Daniel Johnson
# File: linreg.py
# Date: 1/5/2021
# Brief: This script creates uses linear regression model
#        that can be trained and create predictions
'''

import numpy as np
from costs import rmse

class LinRegModel:
    '''
    # @post: A LinRegModel object is created
    # @param: inputs: number of inputs into the network
    #         Theta: a optional parameter to set weights manually
    #         costFunc: a optional perameter to set cost function
    #                   (only one, but added for modularity)
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, Theta: np.ndarray = None, alpha: float = 1e-3, costFunc: str = "MSE") -> None:
        costFunctions = {
            "MSE": rmse
        }
        self.m = len(y)
        self.X = np.hstack( (np.ones( (self.m, 1)), self.scaleFeatures(X)) )
        self.y = y[:, np.newaxis]
        self.thetas = np.random.rand(np.size(X, 1) + 1, 1) if Theta is None else Theta
        self.alpha = alpha
        self.cost = costFunctions[costFunc]

    '''
    # @param: X: the inputs to the model
    # @return: an np.ndarray holding the model's predictions
    '''
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.hstack(( (np.ones( (np.size(X, 0), 1) )), np.dot(self.scaleFeatures(X), self.thetas) )) # X @ theta

    '''
    # @post: the model is trained by updating self.weights
    # @param: maxIters: maximum amount of training iterations
    #         convergenceThreshold: the smallest number of improvement to break learning
    # @return: lists of cost history and thetas history
    '''
    def train(self, maxIters: int = 1000, convergenceThreshold: float = 0) -> list:
        J_Hist = []
        theta_hist = []
        J_Hist.append( self.cost(self.y, self.X @ self.thetas) )
        for i in range(1, maxIters):
            theta_hist.append(self.thetas)
            error = self.X @ self.thetas - self.y
            self.thetas = self.thetas - (self.alpha/self.m) * np.dot(self.X.T, error)

            J_Hist.append(self.cost(self.y, self.X @ self.thetas))
            print("Iteration: ", i, "Cost: ", J_Hist[i])

            if (np.abs(J_Hist[i - 1] - J_Hist[i]) < convergenceThreshold and i > 0):
                print("Training converged at iteration:", i)
                break
        return J_Hist, theta_hist

    '''
    # @param: X: the np.ndarray to be scaled
    # @return: np.ndarray of scaled inputs
    '''
    def scaleFeatures(self, X: np.ndarray) -> np.ndarray:
        return (X - np.mean(X, 0)) / np.std(X, 0)