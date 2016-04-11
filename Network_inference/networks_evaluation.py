"""
Created on Tue Sep 1 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Module to check efficiency of a network inference algorithm
"""

import numpy as np
from random import shuffle, choice

root_mean_squared_error = lambda x, y : np.mean((x-y)**2)**0.5

class PermutationTester():
    """
    Class for testing reestimation of sparse non-negative matrices
    Randomly swaps a certain fraction of the negative vs positive observations
    and tests the reconstruction
    """
    def __init__(self, Y):
        self.nrows, self.ncols = Y.shape
        self.Y = Y
        self.n_positive = np.sum(Y>0)

    def swap_labels(self, fraction_pos):
        """
        Swaps a certain fraction of the positive labels with the
        negative labels
        Uses replacement
        """
        assert fraction_pos > 0 and fraction_pos < 1
        self.n_swap = int(self.n_positive * fraction_pos)
        positive_entries = [(self.Y[i,j], i , j) for i in range(self.nrows)\
                for j in range(self.ncols) if self.Y[i,j] > 0]
        zero_entries = [(self.Y[i,j], i , j) for i in range(self.nrows)\
                for j in range(self.ncols) if not self.Y[i,j] > 0]
        shuffle(positive_entries)
        shuffle(zero_entries)
        self.positive_entries = positive_entries[:self.n_swap]
        self.zero_entries = zero_entries[:self.n_swap]
        self.Y_swapped = self.Y.copy()
        for (v1, i1, j1), (v2, i2, j2) in zip(self.positive_entries, self.zero_entries):
            self.Y_swapped[i1, j1] = v2
            self.Y_swapped[i2, j2] = v1

    def score(self, predictions, metric=root_mean_squared_error):
        """
        Compares the predictions with the origanal values
        """
        return metric(self.Y, predictions)

if __name__ == '__main__':
    Y = np.random.binomial(3, 0.3, size=(100, 50))
    per_tester = PermutationTester(Y)
    per_tester.swap_labels(0.05)
    print(per_tester.score(per_tester.Y_swapped))
