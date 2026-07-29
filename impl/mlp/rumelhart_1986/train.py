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