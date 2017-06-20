from sklearn import svm

X = [[2, 0], [1, 1], [2,3]]  #three point coordinate
y = [0, 0, 1] #lable

clf = svm.SVC(kernel= 'linear')
clf.fit(X,y)
print (clf)
# get support vectors
print (clf.support_vectors_)  #[ [ 1.  1.] [ 2.  3.] ] this two point is support vector point
# get indices of support vectors
print (clf.support_)    # his index is 1 , 2
# get number of support vectors for each class
print (clf.n_support_)

print (clf.predict([2,0]))  #his index is 0