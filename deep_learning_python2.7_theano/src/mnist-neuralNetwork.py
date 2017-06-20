# -*- coding:utf-8 -*-
#测试手写数字进行识别

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper();

print ("training_data")
print (type(training_data))
print (len(training_data))   #50000
print (training_data[0][0].shape)   #(784, 1)
print (training_data[0][1].shape)   #(10, 1)

print ("validation data")
print (len(validation_data)) #10000

print ("test data")
print (len(test_data))   #10000

net = network.Network([784, 8, 10])
net.SGD(training_data, 30, 10, 2.0, test_data=test_data) #神经网络对学习率、训练次数、层数、很敏感



# Epoch 0: 8485 / 10000
# Epoch 1: 8506 / 10000
# Epoch 2: 8737 / 10000
# Epoch 3: 8849 / 10000
# Epoch 4: 8889 / 10000
# Epoch 5: 8900 / 10000
# Epoch 6: 8909 / 10000
# Epoch 7: 8919 / 10000
# Epoch 8: 8982 / 10000
# Epoch 9: 8914 / 10000
# Epoch 10: 8909 / 10000
# Epoch 11: 8956 / 10000
# Epoch 12: 8965 / 10000
# Epoch 13: 8895 / 10000
# Epoch 14: 8911 / 10000
# Epoch 15: 8928 / 10000

#####################################################################################

import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784,30,10], cost=network2.CrossEntropyCost)#cross-entropy作为cost function，进行手写数字进行识别
net.large_weight_initializer()
net.SGD(training_data,30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
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
#
# Epoch 6 training complete
# Accuracy on evaluation data: 9440 / 10000
#
# Epoch 7 training complete
# Accuracy on evaluation data: 9454 / 10000
#
# Epoch 8 training complete
# Accuracy on evaluation data: 9451 / 10000
#
# Epoch 9 training complete
# Accuracy on evaluation data: 9466 / 10000
#
# Epoch 10 training complete
# Accuracy on evaluation data: 9463 / 10000
#
# Epoch 11 training complete
# Accuracy on evaluation data: 9487 / 10000
