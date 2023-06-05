

class Value:

    def __init__(self, data, _children=(), _operation='') -> None:
        self.data = data
        self._prev = set(_children)

    def __repr__(self) -> str:
        return f"Value({self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(data = self.data + other.data, _children = (self, other), _operation = '+')
        return output
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(data = self.data - other.data, _children = (self, other), _operation = '-')
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(data = self.data * other.data, _children = (self, other), _operation = '*')
        return output
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is an instance of the Value class
        output = Value(data = self.data / other.data, _children = (self, other), _operation = '/')
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
    a = Value(2)
    b = Value(-3)
    c = Value(10)

    d = a*b + c
    d = b/a

    print(d.__repr__())


