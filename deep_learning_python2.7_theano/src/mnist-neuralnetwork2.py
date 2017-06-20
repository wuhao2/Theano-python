# -*- coding:utf-8 -*-
"""
import mnist_loader
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784,30,10], cost=network2.CrossEntropyCost)#cross-entropy作为cost function，进行手写数字进行识别
net.large_weight_initializer() #标准正态分布 N(0, 1)初始化权重和偏向
net.SGD(training_data,30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
"""

# the result is :
# Epoch 18 training complete
# Accuracy on evaluation data: 9534 / 10000
# Epoch 0 training complete
# Accuracy on evaluation data: 9154 / 10000
#
# Epoch 1 training complete
# Accuracy on evaluation data: 9286 / 10000
#
# Epoch 2 training complete
# Accuracy on evaluation data: 9312 / 10000
#
# Epoch 3 training complete
# Accuracy on evaluation data: 9345 / 10000
#
# Epoch 4 training complete
# Accuracy on evaluation data: 9425 / 10000
#
# Epoch 5 training complete
# Accuracy on evaluation data: 9443 / 10000

#**************************************************************************
#***************************************************************************
# net.default_weight_initializer() #标准正态分布 N(0, 1/sqrt(n_in))初始化权重和偏向
#新方法初始化权重，测试结果
# Epoch 20 training complete
# Accuracy on evaluation data: 9617 / 10000
# Epoch 23 training complete
# Accuracy on evaluation data: 9579 / 10000
#
# Epoch 24 training complete
# Accuracy on evaluation data: 9597 / 10000
#
# Epoch 25 training complete
# Accuracy on evaluation data: 9593 / 10000
#
# Epoch 26 training complete
# Accuracy on evaluation data: 9573 / 10000
#
# Epoch 27 training complete
# Accuracy on evaluation data: 9585 / 10000


###################################################################################
###################################################################################


# 实现提高版的手写数字识别的方法测试
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network2
net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)

net.SGD(training_data, 30, 10, 0.5, 5.0, evaluation_data = validation_data,
        monitor_evaluation_accuracy=True, monitor_evaluation_cost=True,
        monitor_training_accuracy=True, monitor_training_cost=True)

# Epoch 8 training complete
# Cost on training data: 0.382066881528
# Accuracy on training data: 48222 / 50000
# Cost on evaluation data: 0.921780588632
# Accuracy on evaluation data: 9604 / 10000

# Epoch 12 training complete
# Cost on training data: 0.38370014888
# Accuracy on training data: 48276 / 50000
# Cost on evaluation data: 0.95229508919
# Accuracy on evaluation data: 9585 / 10000
#
# Epoch 13 training complete
# Cost on training data: 0.408049134555
# Accuracy on training data: 48011 / 50000
# Cost on evaluation data: 0.979567719852
# Accuracy on evaluation data: 9535 / 10000
