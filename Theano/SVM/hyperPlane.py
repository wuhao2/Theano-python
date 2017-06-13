# -*- coding:utf-8 -*-
import numpy as np
import pylab as pl
from sklearn import svm
#导入科学计算相关的numpy包和画图要用的pylab包，命名别名np,pl，以及svm

np.random.seed(0)
#seed里面的数字不变每次运行产生的随机数将是同一组
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20
#产生两组随机数，和他们的classlabel标记，+-[2,2]是正态分布保证他们的线性可区分

clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)
#将x,y传入fit建立svm模型
#print(clf.coef_)

w = clf.coef_[0]
#调用coef_取得w值
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
#linspace取得-5到5之间的值用于画线
yy = a * xx - (clf.intercept_[0] / w[1])
#点斜式方程这样就取得了yy的值
#后面就利用xx, 和yy的值画出分界的直线

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
#得到分界线上方和下方与之平行的边际直线的xx和yy后面一并画出

#print "w: ", w
#print "a: ", a

#print "suport_vectors_ :", clf.support_vectors_
#print "clf.coef:  ", clf.coef_

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
#用plot画出3条线，第三个参数设置实线和虚线

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 80, facecolors = 'none')
pl.scatter(x[:, 0], x[:, 1], c = y, cmap = pl.cm.Paired)
#用scatter将支持向量单独圈出来

pl.axis('tight')
pl.show()