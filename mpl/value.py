import math
from typing import List, Set


class Value:

    def __init__(
        self,
        data: float,
        symbol: str = None,
        expression: str = '',
        children: tuple = (),
        operation: str =''
    ) -> None:
        self.data = data
        self.symbol = symbol if symbol else expression
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
        """
        output = self + other
        """
        # ensure other is an instance of the Value class
        other = other if isinstance(other, Value) else Value(other)

        output = Value(
            data = self.data + other.data,
            expression = f"{self.symbol} + {other.symbol}" if self.symbol and other.symbol else '',
            children = (self, other),
            operation = '+'
        )

        def backwards():
            """
            z = x + y
            ∇x = ∇z * dz/dx       dz/dx = 1
            ∇y = ∇z * dz/dy       dz/dy = 1
            """
            self.gradient += 1.0 * output.gradient
            other.gradient += 1.0 * output.gradient
        output.backwards = backwards

        return output

    def __sub__(self, other):
        """
        output = self - other
        """
        # ensure other is an instance of the Value class
        other = other if isinstance(other, Value) else Value(other)

        output = Value(
            data = self.data - other.data,
            expression = f"{self.symbol} - {other.symbol}" if self.symbol and other.symbol else '',
            children = (self, other),
            operation = '-'
        )

        def backwards():
            """
            z = x - y
            ∇x = ∇z * dz/dx       dz/dx =  1
            ∇y = ∇z * dz/dy       dz/dy = -1
            """
            self.gradient += 1.0 * output.gradient
            other.gradient += -1.0 * output.gradient

        output.backwards = backwards
        return output

    def __mul__(self, other):
        """
        output = self * other
        """
        # ensure other is an instance of the Value class
        other = other if isinstance(other, Value) else Value(other)

        output = Value(
            data = self.data * other.data,
            expression = f"{self.symbol} * {other.symbol}" if self.symbol and other.symbol else '',
            children = (self, other),
            operation = '*'
        )

        def backwards():
            """
            z = x * y
            ∇x = ∇z * dz/dx       dz/dx = y
            ∇y = ∇z * dz/dy       dz/dy = x
            """
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient

        output.backwards = backwards
        return output

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        """
        reverse add method

        This functioon is only called when the left operand does not support
        the operation performed by the right operand. For example:

        3 + Value(2) = 3.__add__(Value(2)) = Unimplemented

        Instead of raising an error, the __radd__ method is called which simply
        reverses the operation. In this case, it is equivalent to:

        Value(2) + 3
        """
        # reverse add
        # 3 + Value(2) => Value(2) + 3
        return self + other

    def __rsub__(self, other):
        """see __radd__"""
        return self - other

    def __rmul__(self, other):
        """see __radd__"""
        return self * other

    def __rtruediv__(self, other):
        """see __radd__"""
        return self / other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        output = Value(
            data = self.data**other,
            children=(self,),
            operation=f'**{other}',
            expression=f'{self.symbol}**{self.other}' if self.symbol and other.symbol else '',
        )

        def backwards():
            """
            z = x ^ y
            ∇x = ∇z * dz/dx       dz/dx = y * x^(y-1)
            """
            self.gradient += (other * self.data**(other-1)) * output.gradient

        output.backwards = backwards
        return output

    def tanh(self):
        """
        Perform the tanh operation on the value

        For its definition see:
        https://en.wikipedia.org/wiki/Hyperbolic_functions

        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        output =  Value(
            data= t,
            expression= f"tanh({self.symbol})" if self.symbol else '',
            children=(self, ),
            operation='tanh'
        )

        def backwards():
            self.gradient += (1 - t**2) * output.gradient
        output.backwards = backwards

        return output

    def back_propogate(self):
        """
        Use back propagation to update the gradients of each Value object in the
        computational graph.

        First a topological graph is constructed using dynamic programming. This
        graph contains the furthest children nodes first, and the root node
        last.

        The topological graph is then traversed in reverse order, and the
        gradient is computed.
        """
        self.gradient = 1.0

        def build_topological_graph(
            tree:Value,
            graph:List[Value]=[],
            visited:Set[Value]=set()
        ) -> None:
            """
            Build a topological graph of the tree.

            The topological graph is a list of nodes where the children of a
            node appear before the node itself (i.e. the furthest most children
            appear in the list first, and the parent/root node appears last)
            """
            if tree not in visited:
                visited.add(tree)
                for child in tree.children:
                    build_topological_graph(child, graph, visited)
                graph.append(tree)
            return graph

        for value in reversed(build_topological_graph(self)):
            value.backwards()


def tanh(x:Value) -> Value:
    """
    Assuming `x` is a `Value` object, call its `tanh` method in itself.

    Args:
        x(Value): Value object

    Returns:
        (Value): Value object with the tanh operation applied to it.
    """
    assert isinstance(x, Value), "x should be a Value object"
    return x.tanh()


if __name__ == "__main__":
    from visualization import Visualizer
    visualizer = Visualizer()

    a = Value(2)
    b = Value(-3)
    c = Value(10)

    d = a*b + c
