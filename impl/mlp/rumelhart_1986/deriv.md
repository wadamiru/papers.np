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

### Weight & Bias Initialization
To break symmetry and prevent hidden units from computing identical gradients, weights are initialized to small random values around zero:

$$W^l_{ij} \sim U(-r, r) \quad \text{where } r \approx 0.1 \text{ or } 0.5$$

Biases $b^l$ can be initialized to small random values or set to $0$.

---

## Forward Pass

For layer $l$:

$$z^l = a^{l-1} W^l + b^l$$
$$a^l = \sigma(z^l) = \frac{1}{1 + \exp(-z^l)}$$

---

## Backward Pass — The Generalized Delta Rule

Define the error signal at layer $l$'s net input as:

$$\delta^l \triangleq \frac{\partial E}{\partial z^l}$$

### Output Layer ($L$)
Chain rule through the loss function and sigmoid activation:

$$\frac{\partial E}{\partial a^L} = \frac{a^L - t}{N}$$

$$\delta^L = \frac{\partial E}{\partial a^L} \odot \sigma'(z^L) = \frac{a^L - t}{N} \odot a^L \odot (1 - a^L)$$

> *Note: Uses the standard sigmoid derivative identity: $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = a(1 - a)$.*

### Hidden Layers ($l < L$)
The recursive back-propagation step (Eq. 7 in the paper). The error is pulled backward through the next layer's weights before being multiplied by the local derivative:

$$\frac{\partial E}{\partial a^l} = \delta^{l+1} (W^{l+1})^T$$

$$\delta^l = \frac{\partial E}{\partial a^l} \odot a^l \odot (1 - a^l)$$

> **Key Insight:** This recursion is the central trick of backprop. Because $\delta^l$ only depends on $\delta^{l+1}$ and never on anything further downstream, error signals across an arbitrarily deep stack can be computed in a single backward sweep.

### Parameter Gradients
At every layer $l$:

$$\frac{\partial E}{\partial W^l} = (a^{l-1})^T \delta^l$$

$$\frac{\partial E}{\partial b^l} = \sum_{n=1}^{N} \delta^l_n$$

### Gradient into Previous Layer's Activations
Required to continue the recursion:

$$\frac{\partial E}{\partial a^{l-1}} = \delta^l (W^l)^T$$

---

## Weight Update — Momentum (Eq. 8)

$$v^l(t) = \alpha v^l(t-1) - \epsilon \frac{\partial E}{\partial W^l}$$

$$W^l \leftarrow W^l + v^l(t)$$

* $\alpha = 0.9$ (momentum coefficient used in the paper)
* $\epsilon$ = learning rate (task-dependent)

---

## Sanity Check: XOR

The **XOR** problem is not linearly separable, meaning a single-layer perceptron ([Minsky & Papert, 1969](https://en.wikipedia.org/wiki/Perceptrons_(book))) provably cannot solve it. However, a network with one hidden layer can. 

This serves as the classic proof-of-life for any backpropagation implementation, demonstrating the ability to learn internal representations that linear models cannot.