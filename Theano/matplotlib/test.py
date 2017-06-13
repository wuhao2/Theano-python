# import matplotlib.pyplot as plt
# plt.plot([1,2,3],[1,2,3])
# plt.show()

# import numpy as np
# a=[1,2,3]
# x1=np.array(a)
# x2=np.array(a)
# x1.shape
# x2.shape
# print x1,x2

# c=np.arange(11)
# print c
# print c[1:5]
# print c[:5]
# print c[5:]
# print c[::5]#step is 5


# c=np.random.randint(1,100,10)#generate 10 rand digist between 1 and 100
# print c
# print c[2:5]
# print np.sort(c)
# print np.mean(c)
# print np.min(c)
# print np.max(c)
# print np.median(c)
# print np.var(c)


# c1=np.zeros((2,3)) #init matrix 2*3
# c2=np.arange(10)
# print c1
# print c2

# x=np.loadtxt('000001.csv',delimiter=',',skiprows=1,usecols=(1,4,6),unpack=False)
# # print x
# o,c,v=np.loadtxt('000001.csv',delimiter=',',skiprows=1,usecols=(1,4,6),unpack=True)
# vwap=np.average(c,weights=v)
# print vwap
# print "finish"

import numpy as np
x=np.random.randint(1,100,10)
np.savetxt('testfile.txt',x)
c=np.loadtxt('testfile.txt')
print c
c_sort=np.sort(c)
highest=np.max(c)
lowest=np.min(c)
mean=np.mean(c)
variance=np.var(c)
print "*********************************************************************"
print c_sort,highest,lowest,mean,variance
