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
            input_layer_size:int, 
            hidden_layer_sizes:List[int], 
            output_layer_size:int
        ) -> None:
        layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
        self.layers = self.construct_layers(layer_sizes)

    @staticmethod
    def construct_layers(layer_sizes):
        layers = []
        for i in range(len(layer_sizes)-1):
            n_inputs = layer_sizes[i]
            n_outputs = layer_sizes[i+1]
            layers.append(Layer(n_inputs, n_outputs))
        return layers

    def __call__(self, x):
        """
        Use the previous layer's output for the next layer's input
        """
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def gradient_descent(self, step_size=0.01):
        """
        Update parameters (weights and biases) based on their gradient 
        (partial derrivative of loss function with respect to said parameter)
        """
        for p in self.parameters():
            p.data += -step_size*p.gradient

    def predict(self, x):
        """
        Given input data return the prediction of the model as output data
        """
        y_pred = []
        for xi in x:
            for layer in self.layers:
                xi = layer(xi)
            y_pred.append(xi[0] if len(xi) == 1 else xi)
        return y_pred

    def calculate_loss(self, y_train, y_pred):
        return sum((yt - yp)**2 for yt, yp in zip(y_train, y_pred))  
    
    def train(self, x_train, y_train, thresehold=0.01, max_iterations=1_000, step_size=0.01):
        loss_value = 10**10
        counter = 0
        while loss_value > thresehold:
            if counter > max_iterations:
                print('Exceeded maximum number of iterations')
                break 
            
            y_pred = self.predict(x_train)
            
            loss = self.calculate_loss(y_train, y_pred)
            loss.back_propogate()
            loss_value = loss.data

            self.gradient_descent(step_size)
            counter += 1
            
            print(loss)
         