from typing import Union
import math

class Value:
    """ "The building block of the expression graph"""

    def __init__(
        self,
        data: float,
        _children: tuple["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = data
        # grad to keep track of derivative of the output with respect to this value
        self.grad = (
            0.0  # by default we assume that any input does not impact the output
        )
        # the base case is when values are leaves, in that case they do not do a thing
        # in the general case, it is the function that will do the little piece of chain rule
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # in this value object we can have small atomic operations
    # or big complex abstract operations as we need them
    # the only thing that is important here is that
    # we should know how to differentiate that
    # the local derivative
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union["Value", float]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data + other.data,
            _children=(self, other),
            _op="+",
        )

        # closure to tell the new value how do its local chain rule
        def _backward() -> None:
            # self.grad = 1.0 * out.grad # a bug if self and other are the same
            # other.grad = 1.0 * out.grad
            # add the gradient to previous gradients in case self and other are the same
            # same idea applies to other operations
            # also summing them is how you apply the chain rule in multivariable calculus
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Value", float]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data * other.data,
            _children=(self, other),
            _op="*",
        )

        # closure to tell the new value how do its local chain rule
        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        # not sure I like this interface, but ok
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(data=self.data**other, _children=(self,), _op=f"**{other}")

        def _backward() -> None:
            # n*self.data^{n-1} * out.grad
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: Union["Value", float]) -> "Value":
        return self.__add__(other)

    def __sub__(self, other: Union["Value", float]) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Union["Value", float]) -> "Value":
        return other + (-self)

    # to handle cases in which we have float or int * Value
    def __rmul__(self, other: Union["Value", float]) -> "Value":
        return self * other

    def __truediv__(self, other: Union["Value", float]) -> "Value":
        return self * other**-1

    def __rtruediv__(self, other: Union["Value", float]) -> "Value":
        return other * self**-1

    def backward(self) -> None:
        topo = []
        visited = set()

        def topological_sort(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_sort(child)
                topo.append(v)

        topological_sort(self)
        self.grad = 1.0
        while topo:
            node = topo.pop()
            node._backward()

    def tanh(self) -> "Value":
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(data=t, _children=(self,), _op="tanh")

        # closure to tell the new value how do its local chain rule
        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad  # 1 - tanh^2(x) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        x = self.data
        out = Value(data=math.exp(x), _children=(self,), _op="exp")

        def _backward() -> None:
            self.grad += out.data * out.grad  # e^x * out.grad

        out._backward = _backward
        return out