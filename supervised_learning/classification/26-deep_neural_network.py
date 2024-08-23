#!/usr/bin/env python3
"""Task 26"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """Initializes DeepNeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lix, layer_size in enumerate(layers, 1):
            if type(layer_size) is not int:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layer_size, nx) * np.sqrt(2/nx)
            self.__weights["W{}".format(lix)] = w
            self.__weights["b{}".format(lix)] = np.zeros((layer_size, 1))
            nx = layer_size

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data,
               where nx is number of input features & m is number of examples.

        Returns:
            The output of the neural network and the cache, respectively.
        """
        self.__cache["A0"] = X

        for lay in range(1, self.__L + 1):
            A_prev = self.__cache["A{}".format(lay - 1)]
            Wl = self.__weights["W{}".format(lay)]
            bl = self.__weights["b{}".format(lay)]
            Zl = np.dot(Wl, A_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))
            self.__cache["A{}".format(lay)] = Al

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y: numpy.ndarray with shape (1, m) contains correct labels for
            input data.
            A: numpy.ndarray with shape (1, m) containing activated output
            of neuron for each example.

        Returns:
            The cost.
        """
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X: numpy.ndarray with shape (nx, m) that contains input data,
            where nx is number of input features & m is number of examples.
            Y: numpy.ndarray with shape (1, m) that contains correct
            labels for input data.

        Returns:
            The neuron's prediction and the cost of the network, respectively.
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            Y: numpy.ndarray with shape (1, m) contains correct
            labels for input data.
            cache: dictionary containing all intermediary values of network.
            alpha: learning rate (default value: 0.05).

        Updates:
            The private attribute __weights.
        """
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for icl in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(icl - 1)]
            dW = (1/m) * np.dot(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.__weights["W{}".format(icl)].T, dZ)
            dZ = dA * (A_prev * (1 - A_prev))

            self.__weights["W{}".format(icl)] -= alpha * dW
            self.__weights["b{}".format(icl)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Args:
            X: numpy.ndarray with shape (nx, m) contains input data,
               where nx is number of input features & m is number of examples
            Y: numpy.ndarray with shape (1, m) contains labels for input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: boolean to print information about training
            graph: boolean to graph information about training
            step: number of iterations to print and graph data

        Returns:
            Evaluate training data after training iterations have occurred.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        graphx = []
        graphy = []
        for i in range(0, iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if verbose:
                if i == 0 or i % step == 0:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))
            if graph:
                if i == 0 or i % step == 0:
                    current_cost = self.cost(Y, A)
                    graphy.append(current_cost)
                    graphx.append(i)
                plt.plot(graphx, graphy)
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")
            if verbose or graph:
                if type(step) is not int:
                    raise TypeError("step must be an integer")
                if step <= 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
        if graph:
            plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        '''Save obj as pickle file'''
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        '''Loads pickled DNN obj'''
        try:
            with open(filename, "rb") as file:
                obj = pickle.load(file)
                if isinstance(obj, DeepNeuralNetwork):
                    return obj
                else:
                    return None
        except FileNotFoundError:
            return None

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
