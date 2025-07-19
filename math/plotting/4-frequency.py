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
    bins = np.arange(0, 101, 10)
    clipped_grades = np.clip(student_grades, bins[0], bins[-1])
    plt.hist(clipped_grades, bins=bins, edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(bins[0], bins[-1])
    plt.show()
