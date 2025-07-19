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
    plt.figure(figsize=(6.4, 4.8), dpi=80)
    bins = np.arange(0, 110, 10)
    clipped_grades = np.clip(student_grades, bins[0], bins[-1])
    plt.hist(clipped_grades, bins=bins, edgecolor='black', histtype='bar', align='mid', linewidth=1)
    plt.xlabel('Grades', fontsize=10)
    plt.ylabel('Number of Students', fontsize=10)
    plt.title('Project A', fontsize=12)
    plt.xlim(bins[0], bins[-1])
    plt.ylim(0, None)
    plt.grid(False)  
    plt.tight_layout()
    plt.show()
