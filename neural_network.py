import pandas as pd
import numpy
import math

class NeuralNetwork:
    def __init__(self, d_input, d_output, lr=0.1, max_it=100, activateFunc="step"):
        self.lr = lr
        self.d_input = d_input
        self.d_output = d_output
        self.max_it = max_it
        self._isTrained = False
        self.bias = numpy.full(
            (len(d_output[0]), 1), 0.1, dtype=numpy.float)
        self.weights = numpy.full(
            (len(d_output[0]), len(d_input[0])), 0.0, dtype=numpy.float)
        self.activateFunc = self.step if activateFunc == "step" else self.sigmoid

    # train procedure
    def train(self):
        ERRO = 1
        it = 0
        while it < self.max_it and ERRO > 0.1:
            ERRO = 0
            for index, x in enumerate(self.d_input):
                y = self.evaluate(numpy.asmatrix(x).T)
                e = numpy.asmatrix(self.d_output[index]).T - y
                self.weights = self.weights + self.lr * e * numpy.asmatrix(x)
                self.bias = self.bias + self.lr * e
                ERRO = ERRO + numpy.sum(self.square(e))
            it = it + 1

        print("\nFinal Weights")
        print(self.weights)
        _isTrained = True

    
    # test the value passed
    def evaluate(self, x, print_=False):
        wTimesXPlusB = self.weights.dot(x) + self.bias
        if print_:
            print(wTimesXPlusB)
        return self.activateFunc(wTimesXPlusB)

    def square(self, matrix):
        resp = matrix
        for r, row in enumerate(resp):
            for c, column in enumerate(row):
                resp[r][c] = column**2
        return resp

    def sigmoid(self, value):
        resp = value
        for r, row in enumerate(resp):
            for c, column in enumerate(resp):
                resp[r][c] = 1 / (1 + math.exp(-column))
        return resp
 
    def step(self, value):
        resp = value
        for r, row in enumerate(resp):
            for c, column in enumerate(row):
                if column < 0.5:
                    resp[r][c] = 0
                else:
                    resp[r][c] = 1
        return resp
        


if __name__ == "__main__":
    data = pd.read_csv('iris.data', header=None)

    # amostras
    d_input = data.iloc[:, :-1].to_numpy()

    # rotulos
    d_output = data[4].to_numpy()


    classes = []

    for classification in d_output:
        if classification == 'Iris-setosa':
            classes.append([0.0, 0.0, 1.0])
        elif classification == 'Iris-versicolor':
            classes.append([0.0, 1.0, 0.0])
        elif classification == 'Iris-virginica':
            classes.append([1.0, 0.0, 0.0])

    classes = numpy.array(classes)

    nn = NeuralNetwork(d_input, classes, max_it=20, activateFunc="sigmoid")
    nn.train()
    result = nn.evaluate(numpy.asmatrix([5.1,3.5,1.4,0.2]).T, True)
    
    print("\nFinal Results")
    print(result)
    