#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""4. Frequency
This script plots a histogram of student grades for Project A.
"""


def frequency():
    """Plot a histogram of student grades with specified formatting."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 110, 10)
    student_grades = np.clip(student_grades, 0, 100)
    plt.hist(student_grades, bins=bins, edgecolor='black', align='mid')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.show()
