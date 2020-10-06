import pandas as pd
import numpy
import math
import sys

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
        print("Treinando...")

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

    # Avalia o valor passado

    def evaluate(self, x, print_=False):
        wTimesXPlusB = self.weights.dot(x) + self.bias
        if print_:
            print("\nAvaliação da entrada")
            print(wTimesXPlusB)
        return self.activateFunc(wTimesXPlusB)

    # Eleva ao quadrado todos os elementos da matriz
    def square(self, matrix):
        resp = matrix
        for r, row in enumerate(resp):
            for c, column in enumerate(row):
                resp[r][c] = column**2
        return resp

    # Função de Ativação Sigmoidal
    def sigmoid(self, matrix):
        resp = matrix
        for r, row in enumerate(resp):
            for c, column in enumerate(row):
                resp[r][c] = 1 / (1 + math.exp(-column))

        return self.normalizeByGreater(resp)

    # Normaliza a matriz resultado, transformando o maior valor
    # dentre os elementos o unico não nulo
    def normalizeByGreater(self, matrix):
        resp = matrix
        greater = (0, 0)
        for r, row in enumerate(resp):
            for c, column in enumerate(row):
                if column > resp[greater[0]][greater[1]]:
                    greater = (r, c)

        for r, row in enumerate(resp):
            for c, column in enumerate(row):
                if (r, c) != greater:
                    resp[r][c] = 0
                else:
                    resp[r][c] = 1

        return resp

    # Função de Ativação Degrau
    def step(self, matrix):
        resp = matrix
        for r, row in enumerate(resp):
            for c, column in enumerate(row):
                if column < 0.5:
                    resp[r][c] = 0
                else:
                    resp[r][c] = 1
        return resp


if __name__ == "__main__":

    classification = {
        "Iris-setosa": [0.0, 0.0, 1.0],
        "Iris-versicolor": [0.0, 1.0, 0.0],
        "Iris-virginica": [1.0, 0.0, 0.0]
    }

    data = pd.read_csv('iris.data', header=None)

    # amostras
    d_input = data.iloc[:, :-1].to_numpy()

    # rotulos
    d_output = data[4].to_numpy()

    # classifica as saidas de forma matricial
    classes = []
    for _class in d_output:
        classes.append(classification[_class])

    classes = numpy.array(classes)

    to_classify = []
    print("Insira os valores para a avaliação: num -> [PRESS ENTER]")
    for i in range(4):
        n = input("num {}: ".format(i+1))
        to_classify.append(float(n))

    print("")
    max_it = int(input("Insira o máximo de iterações: dica(200)"))

    print("")
    activateFunc = "step"
    if int(input("Escolha a função de ativação: sigmoid(1) - step(2): ")) == 1:
        activateFunc = "sigmoid"

    # 6.8,2.8,4.8,1.4
    nn = NeuralNetwork(
        d_input, classes, max_it=max_it, activateFunc=activateFunc)
    nn.train()

    result = nn.evaluate(numpy.asmatrix(to_classify).T, True)

    print("\nFinal Results")
    for key, value in classification.items():
        if (result.T == value).all():
            print("{} = {}".format(result.T, key))
            break
