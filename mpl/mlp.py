import random
from typing import Any, List, Tuple
from value import Value, tanh


class Neuron:
    """
    The Neuron class represents a single neuron in a neural network. Each neuron
    has a set of weights and biases that are used to calculate the output of the
    neuron given an input vector.
    """

    activation_functions = {
        'tanh': tanh,
        'linear': lambda x: x,
    }

    def __init__(self, n_input_values:int, activation_function:str='linear') -> None:
        # parameters (weights and biases)
        self.w = [Value(random.uniform(-1, 1), symbol=f'w{i}') for i in range(n_input_values)]
        self.b = Value(random.uniform(-1, 1), symbol='b')

        # activation function
        if activation_function not in self.activation_functions:
            raise ValueError(
                f"Activation function '{activation_function}' not supported. \n"
                f"Supported activation functions: {list(self.activation_functions.keys())}"
            )
        self.f = self.activation_functions[activation_function]

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
        # if s.data> 100 or s.data < -100:
        #     print('Warning: large value')
        return self.f(s)

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

    def __init__(self, n_inputs:int, n_outputs:int, activation_function:str) -> None:
        """
        Args:
            n_inputs (int): number of input values
            n_outputs (int): number of output values
        """
        self.neurons = [Neuron(n_inputs, activation_function) for _ in range(n_outputs)]

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
            hidden_layers: List[Tuple[Any]],
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
        self.layers = self._construct_layers(hidden_layers)

    def _construct_layers(self, hidden_layers: List[Tuple[Any]]) -> List[Layer]:
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
        # determine input/output sizes of model
        nx = len(self.x_train[0]) if isinstance(self.x_train[0], list) else 1
        ny = len(self.y_train[0]) if isinstance(self.y_train[0], list) else 1

        # create hidden layers
        layers = []
        n_inputs = nx
        for n_outputs, activation_function in hidden_layers:
            layers.append(Layer(n_inputs, n_outputs, activation_function))
            n_inputs = n_outputs

        # add output layer (linear activation function)
        layers.append(Layer(n_inputs, ny, 'linear'))

        return layers

    def _process_vector(self, x:List[Value]) -> Value | List[Value]:
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
                y_pred.append( self._process_vector(xi))
            return y_pred
        else:
            y_pred = self._process_vector(x)
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

    def update_parameters(self, step_size:float=0.01) -> None:
        """
        Update parameters (weights and biases) based on their gradient
        (partial derrivative of loss function with respect to said parameter)

        Args:
            step_size (float): scale factor for each step in the gradient
                descent method
        """
        for p in self.parameters:
            # if p.gradient > 50 or p.gradient < -50:
            #     print('Warning: large gradient')
            # if p.data > 50 or p.data < -50:
            #     print('Warning: large gradient')

            p.data += -step_size*p.gradient
            # print()

    def train(self, thresehold=0.01, max_iterations=1_000, step_size=0.01):
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

            # print(f'Iteration: {counter}, Loss: {loss.data}')
            # print(f'Prediction: {y_pred}')
            # print(f'Parameters: {self.parameters}')
            # print(f"Gradients: {[p.gradient for p in self.parameters]}")

            loss_value = loss.data
            counter += 1


if __name__ == "__main__":
    xs = [
        [Value(1., 'x1')],
        [Value(2., 'x2')],
    ]
    ys = [Value(1., 'y1'), Value(2., 'y2')]

    xs = [
        [2.,  3., -1.],
        [3., -1.,  .5],
        [.5,  1.,  1.],
        [1.,  1., -1.],
    ]

    ys = [1., -1., -1., 1.]


    hidden_layers = [(3, 'tanh'), (4, 'tanh')]

    model = MLP(xs, ys, hidden_layers=hidden_layers)
    model.train(thresehold=0.0001, step_size=0.00001, max_iterations=10_000)

    for x in xs:
        y = model.predict(x).data
        print(round(y, 2))
