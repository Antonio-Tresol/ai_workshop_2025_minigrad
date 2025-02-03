import random
from engine import Value

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0
    def parameters() -> list[Value]:
        return []

class Neuron(Module):
    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list) -> Value:
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [param for n in self.neurons for param in n.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]