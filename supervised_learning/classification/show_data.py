#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = os.path.abspath('data')

lib_train = np.load('/home/salma/holberton/holbertonschool-machine_learning/supervised_learning/classification/data/Binary_Train.npz', allow_pickle=True)
X_3D, Y = lib_train['X'], lib_train['Y']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("mygraph.png")