# Multi-Layer Perceptron Neural Network


In order to gain a better understanding of how a neural network works I have built a multi-layer perceptron (MLP) model from scratch. It is very similar to the `torch.autograd` engine, but simplified and more lightweight.

This neural netwrok is able to perform forward propagation in order to determine the output given a set of input data, and backward propgation in order to determine the gradient of each parameter (weight or bias). The neural network calculates a loss function based on the square of the differences in the form of

$$ J = \sum (\hat y_i - y_i)^2 $$

The model is trained by using *gradient descent* to minimize the above loss function. A demonstration can be found in [demo.ipynb](demo.ipynb).

For more information about MLPs, and the implementation in this repo, see [mlp.md](docs/mlp.md)

## Requirements
* python 3.11
* for python packages see `requirements.txt`
* Graphviz (optional for visualizing tree structure used by backpropagation algorithm)
    * Windows: https://graphviz.org/download/
    * MacOS: `brew install graphviz`


## Acknowledgements
This work is based on Andrey Karpathy's [micrograd](https://github.com/karpathy/micrograd), I simply made a similar model in order to gain a better understanding of neural networks.
