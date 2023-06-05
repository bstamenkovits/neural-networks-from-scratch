

class Value:

    def __init__(self, data: float, symbol: str = None, expression: str = None, children: tuple = (), operation: str ='') -> None:
        self.data = data
        self.symbol = symbol if symbol is not None else expression
        self.expression = expression 
        self.operation = operation
        self.gradient = 0

        self.previous = set(children)

    def __repr__(self) -> str:
        return f"Value({self.data})"
    
    def get_label(self):
        if self.expression is not None:
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
        return output
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(
            data = self.data - other.data, 
            expression = f"{self.symbol} - {other.symbol}",
            children = (self, other), 
            operation = '-'
        )
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(
            data = self.data * other.data, 
            expression = f"{self.symbol} * {other.symbol}",
            children = (self, other), 
            operation = '*'
        )
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


if __name__ == "__main__":
    from visualization import Visualizer
    visualizer = Visualizer()
    
    a = Value(2)
    b = Value(-3)
    c = Value(10)

    d = a*b + c
    
    # visualizer.draw_diagram(d)

    # print(d.__repr__())


