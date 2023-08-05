#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
fruits = ['apples', 'bananas', 'oranges', 'peache']
color = ['red', 'yellow', '#ff8000', '#ffe5b4']
plt.bar(range(3), fruit[0], color=color[0], label=fruits[0])
bottom = fruit[0]
for i in range(1, 4):
    plt.bar(range(3), fruit[i], color=color[i], label=fruits[i], bottom=bottom)
    bottom += fruit[i]
plt.xticks(range(3), ['Farrah', 'Fred', 'Felicia'])
plt.ylim((0, 80))
plt.xlabel('People')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')

plt.yticks(np.arange(0, 81, 10))
plt.savefig('my6graph')