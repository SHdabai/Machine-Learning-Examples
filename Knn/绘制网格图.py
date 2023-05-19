
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

'''
np.meshgrid()是Numpy库中用于生成网格点坐标矩阵的函数。在二维平面上，通常使用网格点坐标来表示坐标系中的每个点。

该函数的语法格式如下：

X, Y = np.meshgrid(x, y)
其中，x和y分别为一维数组，表示网格点的x坐标和y坐标；X和Y为二维数组，分别对应x和y的网格点位置矩阵。

具体地说，np.meshgrid()将一维数组x和y转化为二维矩阵X和Y，其中X的每行都是x的复制，Y的每列都是y的复制，这样得到的 X 和 Y 矩阵便可以直接作为横纵坐标轴的值，用于绘制三维曲面图或者二维等高线图。

在数据可视化中，np.meshgrid()通常与 plt.pcolormesh() 或 plt.contourf() 配合使用，用于绘制二维地图、等值线等。

例如，在绘制三维曲面图时，我们需要将三维空间中的点坐标进行离散化，并生成网格点坐标矩阵。此时，可以使用 np.meshgrid() 生成网格点坐标矩阵，然后根据坐标轴上的数值计算每个点的高度值，并将结果可视化。

'''


Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
'''
plt.pcolormesh()是Matplotlib库中可用于绘制坐标系网格的函数之一。它可以用来在二维平面上生成一个伪彩色图像，其主要目的是通过对颜色进行着色来表示数值的大小或密度。

该函数的语法格式如下：

plt.pcolormesh(X, Y, C, cmap=None, norm=None, vmin=None, vmax=None, shading='flat')
其中，X、Y是数组，分别表示网格x轴和y轴的位置坐标；C是与X、Y形状相同的二维数组，表示每个网格的颜色或高度值；cmap是所使用的颜色图表；vmin和vmax是C数组的最小和最大值；shading参数定义着色方式，取值可以是'flat'或'gouraud'。

具体来说，plt.pcolormesh()将X、Y和C转化为网格，并根据其中每个元素的值来着色，在坐标系中生成一个伪彩色二维图像。

在数据可视化中，plt.pcolormesh()常用于绘制热力图、接收器操作特征曲线（ROC曲线）和精确度-召回率（PR曲线）等图像。


'''




# 绘制训练点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('3-Class classification (k = %i, weights = "%s")' % (n_neighbors, weights))

plt.show()