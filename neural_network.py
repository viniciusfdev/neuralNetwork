import pandas as pd
import numpy
import math


class NeuralNetwork:
    def __init__(self, d_input, d_output, bias=0.1, lr=0.1, max_it=10):
        self.d_input = d_input
        self.d_output = d_output
        self.bias = bias
        self.lr = lr
        self.max_it = max_it
        self._isTrained = False

        # TODO: initialize the weights
        # self.initWeights = [for ]

    # test the value passed
    def evaluate(self, array):
        pass

    # train procedure
    def train(self):
        # TODO: implement procedure
        _isTrained = True

    def sigmoid(self, value):
        return 1 / (1 + math.exp(-value))

    def step(self, value):
        if value >= 0.5:
            return 1
        return 0


if __name__ == "__main__":
    data = pd.read_csv('iris.data', header=None)

    # amostras
    d_input = data.iloc[:, :-1].to_numpy()

    # rotulos
    d_output = data[4].to_numpy()

    nn = NeuralNetwork(d_input, d_output)
    nn.train()
    nn.evaluate([])
