import math 


class Value:

    def __init__(self, data: float, symbol: str = None, expression: str = None, children: tuple = (), operation: str ='') -> None:
        self.data = data
        self.symbol = symbol if symbol is not None else expression
        self.expression = expression 
        self.operation = operation
        self.backwards = lambda: None
        self.gradient = 0

        self.children = set(children)

    def __repr__(self) -> str:
        return f"Value({self.data})"
    
    def get_label(self):
        if self.expression:
            label = f"{self.symbol} = {self.expression}"
            if self.expression == self.symbol:
                label = f"{self.expression}"
        else:
            label = self.symbol
        return label
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(
            data = self.data + other.data, 
            expression = f"{self.symbol} + {other.symbol}",
            children = (self, other), 
            operation = '+'
        )

        def func():
            self.gradient += 1.0 * output.gradient
            other.gradient += 1.0 * output.gradient
        output.backwards = func
        
        return output
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(
            data = self.data - other.data, 
            expression = f"{self.symbol} - {other.symbol}",
            children = (self, other), 
            operation = '-'
        )

        def func():
            self.gradient += 1.0 * output.gradient
            other.gradient += -1.0 * output.gradient
        output.backwards = func
        
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(
            data = self.data * other.data, 
            expression = f"{self.symbol} * {other.symbol}",
            children = (self, other), 
            operation = '*'
        )

        def func():
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient
        output.backwards = func
        
        return output
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(
            data = self.data / other.data, 
            expression = f"{self.symbol} / {other.symbol}",
            children = (self, other), 
            operation = '/'
        )
        return output
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return self - other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return self / other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        output = Value(self.data**other, (self,), f'**{other}')

        def func():
            self.gradient += (other * self.data**(other-1)) * output.gradient
        output.backwards = func

        return output

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        output =  Value(
            data= (math.exp(2*x) - 1) / (math.exp(2*x) + 1),
            expression= f"tanh({self.symbol})",
            children=(self, ),
            operation='tanh'
        )

        def func():
            self.gradient += (1 - t**2) * output.gradient
        output.backwards = func
        
        return output        

    def back_propogate(self):
        self.gradient = 1.0

        def build_topological_graph(tree, graph=[], visited=set()):
            if tree not in visited:
                visited.add(tree)
                for child in tree.children:
                    build_topological_graph(child, graph, visited)
                graph.append(tree)
            return graph
        
        for value in reversed(build_topological_graph(self)):
            value.backwards()
            

if __name__ == "__main__":
    from visualization import Visualizer
    visualizer = Visualizer()
    
    a = Value(2)
    b = Value(-3)
    c = Value(10)

    d = a*b + c
