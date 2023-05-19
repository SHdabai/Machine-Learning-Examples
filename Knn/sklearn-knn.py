# -*- coding: UTF-8 -*-

'''
飞行常客里程数、玩视频游戏所耗时间百分比、每周消费的冰淇淋公升数


'''

'''
1. `准备数据`：将已知类别的数据集按照一定规则进行划分，通常把其中的一部分作为训练集，另一部分作为测试集。

2. `计算距离`：对于每一个`待分类`的样本，计算它与训练集中所有样本之间的距离。通常使用`欧氏距离`（Euclidean Distance）或`曼哈顿距离`（Manhattan Distance）等度量方式来进行距离计算。
常见的`欧式距离`：
```python
d(p, q) = sqrt((p1 - q1)^2 + (p2 - q2)^2 + ... + (pn - qn)^2)
```
其中，p1, p2, ..., pn和q1, q2, ..., qn分别表示向量p和向量q在各自的第1个维度到第n个维度上的坐标值。sqrt表示计算平方根。


3. 选择`K`值：根据预设的K值，选取距离该样本`最近的K个样本`。

4. `进行预测`：根据K个最近的样本的类别，通过投票决定该样本所属的类别。通常采用多数表决（Majority Vote）的方式，即选择`出现次数最多`的类别作为预测结果。

5. `评估模型`：使用测试集对模型进行评估，通常使用错误率（Error Rate）或精确度（Accuracy）等指标来衡量模型的性能。


'''

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import numpy as np


def select_k():

    #todo 加载数据
    data = pd.read_csv('knnData.csv')
    x = data.iloc[:,0:3]
    y = data.iloc[:,-1:]
    print(x)
    k_accuracy = []
    for i in range(1,31):
        knn = KNeighborsClassifier(n_neighbors=i)
        socres = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
        k_accuracy.append(socres.mean())
        '''
        在进行交叉验证时，我们会对模型在不同的“折”上的表现进行评估，并计算所有“折”的得分的平均值。因此，这里使用scores.mean()将模型在每个“折”上的得分求平均，得到一个k值对应的平均准确率。
        cross_val_score函数默认返回每个“折”上的得分（即一个长度为10的数组），而这里我们需要的是该k值下的平均准确率，因此使用scores.mean()将数组中的得分求平均。
        具体来说，scores.mean()计算的是模型在10折交叉验证中所有“折”上得分的平均值，作为当前k值的平均准确率。  
        '''
    plt.plot(range(1,31),k_accuracy)
    plt.xlabel('Value of k for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


h = .02 # 网格步长

data = pd.read_csv('knnData.csv')
#进行特征值的选择
x = data.iloc[:,0:3]
y = data.iloc[:,-1:]
# 创建色彩图
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

'''
weights（权重）：最普遍的 KNN 算法无论距离如何，权重都一样，但有时候我们想搞点特殊化，比如距离更近的点让它更加重要。这时候就需要 weight 这个参数了，这个参数有三个可选参数的值，决定了如何分配权重。参数选项如下：
• ‘uniform’：不管远近权重都一样，就是最普通的 KNN 算法的形式。
• ‘distance’：权重和距离成反比，距离预测目标越近具有越高的权重。
• 自定义函数：自定义一个函数，根据输入的坐标值返回对应的权重，达到自定义权重的目的。
'''
#todo 分别在两种权重情况下进行对比

for weights in ["uniform","distance"]:

    #todo 第一步：创建knn分类器
    knn = neighbors.KNeighborsClassifier(n_neighbors = 18,weights=weights)
    knn.fit(X_train,y_train)

    #todo 第二步: 用测试集进行预测，并计算准确率
    y_pred = knn.predict(X_test)
    print(y_pred)
    # accuracy = np.mean(y_pred == y_test)
    # print(f"Accuracy: {accuracy:.2f}")

    #使用模型预测新的数据点
    X_new = np.array([[38344, 3.0, 0.1343], [3436, 3.0, 4.0], [7, 9.0, 6.0]])
    y_new = knn.predict(X_new)  # 预测新的数据点类别

    print(f"New data point {X_new[0]} belongs to class {[y_new[0]]}")
    print(f"New data point {X_new[1]} belongs to class {[y_new[1]]}")
    print(f"New data point {X_new[2]} belongs to class {[y_new[2]]}")






