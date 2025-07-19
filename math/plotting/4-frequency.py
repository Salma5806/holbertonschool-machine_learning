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
    
    # Clip grades so no values outside 0-100 bins
    student_grades = np.clip(student_grades, 0, 100)
    
    plt.figure(figsize=(6.4, 4.8))
    
    # bins edges from 0 to 100 every 10 units, add a small epsilon to include 100 exactly
    bins = np.arange(0, 101, 10)
    
    plt.hist(student_grades, bins=bins, edgecolor='black', density=False)
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.show()
