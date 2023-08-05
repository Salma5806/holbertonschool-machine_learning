#!/usr/bin/env python3
""" plots x, y as line graph """
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y)
plt.title("Exponential Decay of C-14")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.xlim((0, 28650))
plt.yscale("log")
plt.savefig("my2graph.png")