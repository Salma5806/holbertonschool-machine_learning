#!/usr/bin/env python3
""" Script to evaluate the prediction of a neuron """
import numpy as np


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """Initialize neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ return weight """
        return self.__W

    @property
    def b(self):
        """ return biais"""
        return self.__b

    @property
    def A(self):
        """return output"""
        return self.__A

    def forward_prop(self, X):
        """Function of forward propagation """
        Z = self.__b + np.dot(self.__W, X)
        sigmoid = 1 / (1 + np.exp(-Z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """ cost function using binary cross entropy """

        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """ evalute the neuron’s predictions  """
        self.forward_prop(X)
        c = self.cost(Y, self.__A)
        p = np.where(self.__A >= 0.5, 1, 0)
        return p, c

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ calculate the gradient descent """
        m = X.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ defines a single neuron performing binary  """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
