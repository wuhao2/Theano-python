# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:49:09 2017

@author: Alien
"""

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

# 读取文件
with open(r'/root/deeplearning/decisionTree/allElectronicsData.csv', 'r') as allElectronicsData:
    allElectronicsData = open(r'/root/deeplearning/decisionTree/allElectronicsData.csv', 'r')
    reader = csv.reader(allElectronicsData)
    headers = next(reader)
    print(headers)    #['RID', 'age', 'income', 'student', 'credit_rating', 'Class:buy_computer']

featureList = []
labelList = []
for row in reader:
    labelList.append(row[len(row) - 1])  #define the last row as labelList
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)    #[{'credit_rating': 'fair', 'age': 'youth', 'student': 'no', 'income': 'high'}, {'credit_rating': 'excellent', 'age': 'youth', 'student': 'no', 'income': 'high'}, {'credit_rating': 'fair', 'age': 'middle_age
allElectronicsData.close()

# 转化数据为sklearn要求的数据
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:" + str(dummyX))
print(vec.get_feature_names())
print("labelList:" + str(labelList))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:" + str(dummyY))


## 决策树处理
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf:" + str(clf))


## 将决策树输出位dot文件
with open("/root/deeplearning/decisionTree/allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)


oneRowX = dummyX[0, :]
print("oneRowX:" + str(oneRowX))#oneRowX:[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]

# ## 给定新的数据进行预测
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX:" + str(newRowX)) #newRowX:[ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]

predictedY = clf.predict(newRowX)
print("result predictedY:" + str(predictedY))#result predictedY:[1]  means buy_computer