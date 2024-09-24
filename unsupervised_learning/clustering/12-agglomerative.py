#!/usr/bin/env python3
"""Task 12"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Placeholder"""
    linkage_matrix = scipy.cluster.hierarchy.ward(X)
    fcluster = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, dist, criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix, color_threshold=dist)
    plt.show()
    return fcluster
