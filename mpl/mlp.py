import random
from typing import Any, List
from value import Value, tanh


class Neuron:
    """
    The Neuron class represents a single neuron in a neural network. Each neuron
    has a set of weights and biases that are used to calculate the output of the
    neuron given an input vector.
    """

    def __init__(self, n_input_values:int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_input_values)]
        self.b = Value(random.uniform(-1, 1))

    def __repr__(self):
        return f"Neuron({len(self.w)})"

    def __call__(self, x: List[float]) -> Value:
        """
        Given an input vector x, calculate the output of the neuron using the
        following formula:

        output = tanh (Σ(x_i * w_i) + b)

        Args:
            x (List[float]): input vector

        Returns:
            Value: output of the neuron
        """
        assert len(x) == len(self.w), "inputs and weights are of different sizes"
        s = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return tanh(s)

    @property
    def parameters(self) -> List[Value]:
        """All the weights and biases of the neuron"""
        return self.w + [self.b]


class Layer:
    """
    A Layer object represents a set of neurons in a neural network. Each neuron
    in the layer has the same number of input values, and the output of each
    neuron in the layer is used as input for the next layer in the neural
    network.
    """

    def __init__(self, n_inputs:int, n_outputs:int) -> None:
        """
        Args:
            n_inputs (int): number of input values
            n_outputs (int): number of output values
        """
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __repr__(self):
        return f"Layer({len(self.neurons)})"

    def __call__(self, x: List[float]) -> List[Value]:
        """
        Given an input vector x, calculate the output of each neuron in the
        layer, and return a list of the output values (output vector).

        Args:
            x (List[float]): input vector

        Returns:
            List[Value]: output vector
        """
        return [neuron(x) for neuron in self.neurons]

    @property
    def parameters(self):
        """All the weights and biases of the neurons in the layer"""
        return [p for neuron in self.neurons for p in neuron.parameters]


class MLP:

    def __init__(
            self,
            x_train:List[float] | List[List[float]],
            y_train:List[float] | List[List[float]],
            hidden_layer_sizes: List[int],
        ) -> None:
        """
        Initialize the Multi-Layer Perceptron (MLP) model with the input data
        and hidden layer sizes.

        x_train is the input data, and y_train the expected output data. Both
        the input and output data can either be a single vector (List[float]) or
        a list of vectors (List[List[float]]).

        The hidden layers can be configured by providing a list of integers.
        Each integer in the list represents the number of neurons in that layer.

        Args:
            x_train (List[float] | List[List]): input data
            y_train List[float] | List[List]): output data
            hidden_layer_sizes (List[int]): list of integers representing the
                size of each hidden layer in the neural network
        """
        self.x_train = x_train
        self.y_train = y_train
        self.layers = self._construct_layers(hidden_layer_sizes)

    def _construct_layers(self, hidden_layer_sizes: List[int]) -> List[Layer]:
        """
        Construct the layers of the neural network based on the input data.

        First the size of each layer is determined by counting the number of
        elements in the input data, the number of elements in the output data,
        and the hidden layer sizes.


        Then the layer objects are created. Each object represents a set of
        transformations that are applied to the input data for a given layer.

        Args:
            hidden_layer_sizes (List[int]): list of integers representing the
                size of each hidden layer in the neural network

        Returns:
            List[Layer]: list of layers in the neural network
        """
        # check that x_train and y_train have the same dimensions
        xdim = len(self.x_train)
        ydim = len(self.y_train)

        if not xdim == ydim:
            raise ValueError("x_train and y_train should have the same dimensions")

        # determine layer input/output sizes
        nx = len(self.x_train[0]) if isinstance(self.x_train[0], list) else 1
        ny = len(self.y_train[0]) if isinstance(self.y_train[0], list) else 1
        sizes = [nx] + hidden_layer_sizes + [ny]

        # construct layer objects
        layers = []
        for i in range(len(sizes)-1):
            n_inputs = sizes[i]
            n_outputs = sizes[i+1]
            layers.append(Layer(n_inputs, n_outputs))
        return layers

    def _processs_vector(self, x:List[Value]) -> Value | List[Value]:
        """
        Take an input vector `x` and return the neural network output vector by
        going through each layer and using the previous layer's output for the
        next layer's input
        """
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    @property
    def parameters(self):
        """
        Obtain all the parameters (weights and biases) of the neural network
        """
        return [p for layer in self.layers for p in layer.parameters]

    def update_parameters(self, step_size:float=0.01) -> None:
        """
        Update parameters (weights and biases) based on their gradient
        (partial derrivative of loss function with respect to said parameter)

        Args:
            step_size (float): scale factor for each step in the gradient
                descent method
        """
        for p in self.parameters:
            p.data += -step_size*p.gradient

    def predict(self, x:List[float] | List[List[float]]) -> List[float]:
        """
        Given input data x return the prediction y_pred of the model.

        If x is a list of vectors, return a list of predictions for each vector
        in x. Otherwise, return a single prediction for the single input vector
        x.

        Args:
            x (List[float] | List[List[float]]): input data
        """
        if isinstance(x[0], list):
            y_pred = []
            for xi in x:
                y_pred.append( self._processs_vector(xi))
            return y_pred
        else:
            y_pred = self._processs_vector(x)
            return y_pred

    def calculate_loss(self, y_train, y_pred):
        """
        Calculate the loss function. In this case we are using the Mean Squared
        Error (MSE) loss function:

        J = ∑(yt_i - yp_i)^2
        """
        return sum((yt - yp)**2 for yt, yp in zip(y_train, y_pred))

    def update_gradients(self, loss:Value) -> None:
        """
        Use backpropagation to calculate the gradient of the loss function with
        respect to each parameter (weights and biases) in the neural network.
        """
        # reset gradients
        for p in self.parameters:
            p.grad = 0
        loss.back_propogate()

    def train(self, thresehold=0.001, max_iterations=1_000, step_size=0.01):
        loss_value = 10**10
        counter = 0
        while loss_value > thresehold:
            if counter > max_iterations:
                print('Exceeded maximum number of iterations')
                break

            # calculate the prediction
            y_pred = self.predict(self.x_train)

            # calculate loss function J (Mean Squared Error)
            loss = self.calculate_loss(self.y_train, y_pred)

            # use gradient descent to update the parameters
            self.update_gradients(loss)
            self.update_parameters(step_size)

            loss_value = loss.data
            counter += 1


if __name__ == "__main__":
    xs = [
        [2.,  3., -1.],
        [3., -1.,  .5],
        [.5,  1.,  1.],
        [1.,  1., -1.],
    ]

    ys = [1., -1., -1., 1.]

    model = MLP(xs, ys, hidden_layer_sizes=[4, 4])
    # print()
    model.train()

    print(model.parameters)

    for x in xs:
        y = model.predict(x).data
        print(round(y, 2))
