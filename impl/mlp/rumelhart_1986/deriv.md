# Rumelhart, Hinton & Williams (1986) - Derivations
## *Learning Representations by Back-Propagating Errors*

---

## Model Setup

Consider a network with $L$ layers. For each layer $l \in \{1, \dots, L\}$:

$$z^l = a^{l-1} W^l + b^l$$
$$a^l = \sigma(z^l)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function, $a^0 = x$ (input), and $a^L$ is the final network output.

* **Batch-First Convention:** $x$ has shape $(N, D)$, with one row per example across $N$ total samples.
* **Loss Function (Sum-of-Squares, Eq. 1, Batch-Averaged):**

$$E = \frac{1}{2N} \sum_n \sum_c (t_{nc} - a^L_{nc})^2$$

---