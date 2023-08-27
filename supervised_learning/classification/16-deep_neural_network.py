#!/usr/bin/env python3
""" defines DeepNeuralNetwork class that defines a deep neural network  """


import numpy as np

class DeepNeuralNetwork:
    """class that represents a deep neural network"""

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        
        for l in range(1, self.__L + 1):
            if type(layers[l - 1]) is not int or layers[l - 1] <= 0:
                raise TypeError("layers must be a list of positive integers")
            
            if l == 1:
                self.__weights['W' + str(l)] = np.random.randn(layers[l - 1], nx) * np.sqrt(2/nx)
            else:
                self.__weights['W' + str(l)] = np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2/layers[l - 2])
            
            self.__weights['b' + str(l)] = np.zeros((layers[l - 1], 1))

