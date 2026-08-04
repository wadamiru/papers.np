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
    """linear + sigmoid layers stacked. layer_sizes e.g. [2,4,1]"""

    def __init__(self, l_sz, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for nin, nout in zip(l_sz[:-1], l_sz[1:]):
            self.W.append(rng.uniform(-0.3, 0.3, size=(nin, nout)))
            self.b.append(rng.uniform(-0.3, -0.3, size=(nout,)))
        # momentum velocity buffers, one per parameter
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        # layer activations cache for backward()
        self._a = None  # a[0] = input

    def forward(self, x):
        self._a = [x]
        for W, b in zip(self.W, self.b):
            z = self._a[-1] @ W + b
            self._a.append(sigmoid(z))
        return self._a[-1]

    def backward(self, target):
        """Backprop the generalized delta rule (deriv.md). Returns
        (dW list, db list) matching self.W / self.b order."""
        n = target.shape[0]
        a_out = self._a[-1]

        # delta^L = dE/dz^L = dE/da^L * da^L/dz^L
        delta = (a_out - target) / n * a_out * (1 - a_out)

        dW, db = [None] * len(self.W), [None] * len(self.b)
        for l in reversed(range(len(self.W))): # L-1 to 0
            # act feeding into layer l (a^(l-1))
            a_prev = self._a[l]
            
            dW[l] = a_prev.T @ delta
            db[l] = delta.sum(axis=0)

            if l > 0:
                # propagate error to previous layer's activations, then
                # through that layer's own sigmoid derivative to get its delta
                da_prev = delta @ self.W[l].T            # dE/da^l = delta^l+1 @ W^l+1.T
                delta = da_prev * a_prev * (1 - a_prev)  # delta^l = dE/da^l * a^l * (1-a^l)
        return dW, db