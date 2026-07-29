"""
Rumelhart, Hinton & Williams (1986) - Learning representations by back-propagating errors.

Pure numpy. Every gradient is hand-derived (deriv.md).
Convention: batch-first arrays, (x @ W). deriv.md for
the full generalised delta rule derivation.

Run:
    python train.py              # trains on XOR
    python train.py --gradcheck  # verifies analytic grads
"""

import argparse
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class MLP:
    """
    linear + sigmoid layers stacked. layer_sizes e.g. [2,4,1]
    Weight init.: 
    """

    def __init__(self, l_sz, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for nin, nout in zip(l_sz[:-1], l_sz[1:]):
            