# -*- coding:utf-8 -*-
# 3层
# 隐藏层: 100个神经元
# 训练60个epochs
# 学习率 = 0.1
# mini-batch size: 10

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

"""
net = Network([FullyConnectedLayer(n_in=784, n_out=100),#隐藏层神经元个数100
             SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1,
            validation_data, test_data)

# 结果: 97.85 accuracy  (上节课98.04%
# 这次: 没有regularization, 上次有
# 这次: softmax 上次: sigmoid + cross-entropy
"""
##################################################################################
"""
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*12*12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
# 准确率: 98.78 比上次有显著提高
"""
######################################################################################

"""
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1,
            validation_data, test_data)
# 准确率: 99.06% (再一次刷新)
"""
########################################################################################


# 用Rectified Linear Units代替sigmoid:
# f(z) = max(0, z)
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=network3.ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)
# 准确率: 99.23 比之前用sigmoid函数的99.06%稍有提高

###############################################################################################

"""
# 库大训练集: 每个图像向上,下,左,右移动一个像素
# 总训练集: 50,000 * 5 = 250,000
# $ python expand_mnist.py
expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(expanded_training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)
 # 结果: 99.37%
"""

##########################################################################################
"""
# 加入一个100个神经元的隐藏层在fully-connected层:
expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=network3.ReLU),
        FullyConnectedLayer(n_in=100, n_out=100, activation_fn=network3.ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(expanded_training_data, 60, mini_batch_size, 0.03,
            validation_data, test_data, lmbda=0.1)
# 结果: 99.43%, 并没有大的提高
# 有可能overfit
"""


###########################################################################################################


# 加上dropout到最后一个fully-connected层:
expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=network3.ReLU),
        FullyConnectedLayer(
            n_in=40*4*4, n_out=1000, activation_fn=network3.ReLU, p_dropout=0.5),
        FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=network3.ReLU, p_dropout=0.5),
# 只对最后一层进行dropout
# CNN本身的convolution层对于overfitting有防止作用: 共享的权重造成convolution filter强迫对于整个图像进行学习
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)
net.SGD(expanded_training_data, 40, mini_batch_size, 0.03,
            validation_data, test_data)
# 结果: 99.60% 显著提高