import random
from typing import Any, List
from value import Value


class Neuron:

    def __init__(self, n_input_values) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_input_values)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x) -> Value:
        """
        Calculate the neuron output using the following formula for every x_i 
        in x 

        tanh (Σ x_i * w_i + b) 
        """
        assert len(x) == len(self.w), "inputs and weights are of different sizes"
        return ( sum((wi*xi for wi, xi in zip(self.w, x)), self.b) ).tanh()
    
    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, n_inputs, n_outputs) -> None:
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        """
        Calculate the output for each Neuron in this layer and return them as a 
        list. 
        """
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        


class MLP:
    
    def __init__(
            self, 
            x_train, 
            y_train, 
            hidden_layer_sizes, 
        ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.layers = self.construct_layers(hidden_layer_sizes)

    def construct_layers(self, hidden_layer_sizes):
        xdim = len(self.x_train)
        ydim = len(self.y_train)
        assert xdim == ydim, "x_train and y_train should have the same dimensions"
        
        nx = len(self.x_train[0]) if isinstance(self.x_train[0], list) else 1
        ny = len(self.y_train[0]) if isinstance(self.y_train[0], list) else 1
        layer_sizes = [nx] + hidden_layer_sizes + [ny]

        layers = []
        for i in range(len(layer_sizes)-1):
            n_inputs = layer_sizes[i]
            n_outputs = layer_sizes[i+1]
            layers.append(Layer(n_inputs, n_outputs))
        return layers

    def processs_vector(self, x:List[Value]):
        """
        Take an input vector `x` and return the neural network output vector by 
        going through each layer and using the previous layer's output for the 
        next layer's input
        """
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        """
        Obtain all the parameters (weights and biases) of the neural network
        """
        return [p for layer in self.layers for p in layer.parameters()]
    
    def perform_gradient_descent(self, step_size:float=0.01):
        """
        Update parameters (weights and biases) based on their gradient 
        (partial derrivative of loss function with respect to said parameter)

        Parameteres:
            - step_size (float): scale factor for each step in the gradient 
            descent method
        """
        for p in self.parameters():
            p.data += -step_size*p.gradient

    def predict(self, x):
        """
        Given input data return the prediction of the model as output data
        """
        if isinstance(x[0], list):
            y_pred = []
            for xi in x:
                y_pred.append( self.processs_vector(xi))
            return y_pred
        else:
            y_pred = self.processs_vector(x)
            return y_pred

    def calculate_loss(self, y_train, y_pred):
        return sum((yt - yp)**2 for yt, yp in zip(y_train, y_pred))  
    
    def perform_backward_propagation(self, loss):
        for p in self.parameters():
            p.grad = 0
        loss.back_propogate()

    
    def train(self, thresehold=0.01, max_iterations=1_000, step_size=0.01):
        loss_value = 10**10
        counter = 0
        while loss_value > thresehold:
            if counter > max_iterations:
                print('Exceeded maximum number of iterations')
                break 
            
            y_pred = self.predict(self.x_train)
            loss = self.calculate_loss(self.y_train, y_pred)
            self.perform_backward_propagation(loss)
            self.perform_gradient_descent(step_size)
            
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
    model.train()

    for x in xs:
        print(model.predict(x))
        