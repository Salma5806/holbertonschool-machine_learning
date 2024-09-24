#!/usr/bin/env python3
"""Task 10"""

import sklearn.cluster


def kmeans(X, k):
    """Placeholder"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
