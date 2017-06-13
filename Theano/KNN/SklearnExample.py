from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()
print iris

#build model
knn.fit(iris.data, iris.target)
#pridict classcification
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print "reslut:" , predictedLabel  #reslut: [0]