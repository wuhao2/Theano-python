# -*- coding:utf-8 -*-
import numpy as np


# 定义两个非线性转化函数sigmiod
def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):  # 导数
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []  # 权重
        for i in range(1, len(layers) - 1):  # layer 表示层数
            self.weights.append(
                (2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)  # 初始化，第i-1层与第i层之间的权重
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):  # X表示一个矩阵，行代表实例（特征向量），列代表特征（维度）
        X = np.atleast_2d(X)  # 至少是二维的数据，转化为科学计算包numpy数据类型
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # 在所有的行中随机抽取一行i
            a = [X[i]]

            for l in range(len(self.weights)):  # going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l])))  # 加权求和dot，非线性转化activation---正向更新

            error = y[i] - a[-1]  # Computer the error at the top layer 计算class lable与预测值的误差
            deltas = [
                error * self.activation_deriv(a[-1])]  # For output layer, Err calculation (delta is updated error)

            # Staring backprobagation反向更新
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


