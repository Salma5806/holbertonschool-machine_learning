#!/usr/bin/env python3
""" Task 1"""
import numpy as np


class Node:
    """A class representing a node in a decision tree"""

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """Initializes a Node with the given parameters"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculates the maximum depth of the subtree rooted at this node"""
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = self.depth
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes in the subtree rooted at this node"""
        if self.is_leaf:
            return 1
        if self.left_child:
            left_count = self.left_child.count_nodes_below(only_leaves)
        else:
            left_count = 0
        if self.right_child:
            right_count = self.right_child.count_nodes_below(only_leaves)
        else:
            right_count = 0
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count


class Leaf(Node):
    """
    A class representing a leaf node in a decision tree, inheriting from Node"""

    def __init__(self, value, depth=None):
        """Initializes a Leaf with the given parameters"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Counts the number of nodes in the subtree rooted at this leaf"""
        return 1


class Decision_Tree():
    """A class representing a decision tree"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes a Decision_Tree with the given parameters"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts the number of nodes in the decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)
