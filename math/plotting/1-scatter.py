#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def scatter():
    """
    scatter plot
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, color='magenta')  # magenta points
    plt.xlabel('Height (in)')           # x-axis label
    plt.ylabel('Weight (lbs)')          # y-axis label
    plt.title("Men's Height vs Weight") # title
    plt.show()
