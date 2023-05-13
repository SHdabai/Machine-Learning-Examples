# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import StandardScaler  #进行数据标准化处理
from sklearn.metrics import silhouette_score      #进行轮廓系数计算





data = pd.read_csv('./data.csv',error_bad_lines=False,warn_bad_lines=False)



def kemans():

    # print(data)

    # 提取特征值 提取除了名字以外所有列信息
    X = data.loc[:, ['calories', 'sodium', 'alcohol', 'cost']]
    # print(X)

    #使用聚类  默认使用的是 init='k-means++' 进行初始化

    #todo 首先，创建了一个KMeans对象kmean，并设置聚类中心的数量为3个。
    kmean = KMeans(n_clusters=2, init='k-means++')


    #todo 增加数据标准化
    std = StandardScaler()
    x = std.fit_transform(X)


    #todo 然后，使用数据X对KMeans模型进行拟合（fit）操作，得到已经训练好的模型km。
    km = kmean.fit(x)


    #todo 将每个样本点所属的簇(cluster)标签（即聚类结果）添加到原始数据集data中，
    # 并存储在一个名为"cluster"的新列中。这一步操作涉及到km.labels_属性，可以得到每个数据点分配给哪个簇。
    data['cluster'] = km.labels_
    # print(data)
    '''
    km.labels_:是 KMeans 算法对每个样本点进行聚类后得到的簇标签。在 KMeans 模型中，每个簇都有一个编号（从0开始）
       name  calories  sodium  alcohol  cost  cluster
    0    燕京       144      15      4.5  0.43        1
    1    青岛       151      16      5.5  0.65        2
    2    崂山       157      17      4.5  0.56        2
    3    南阳       170      18      4.5  0.98        2
    4    郑州       152      19      4.6  0.54        2
    5    花旗       145      15      4.8  0.23        1
    6    乐视       125      16      4.6  0.65        0
    '''

    #todo 最后：使用groupby方法按照"cluster"列进行分组，并计算每个簇的均值，从而得到聚类中心。
    # 这些聚类中心被存储在centers变量中。

    # centers = data.groupby('cluster').mean().rest_index()
    centers = data.groupby('cluster').mean()
    # print(centers)
    '''
               calories     sodium   alcohol      cost
    cluster                                           
    0         88.250000  17.500000  4.625000  0.507500
    1        160.250000  19.375000  5.062500  0.583750
    2        135.666667  15.666667  4.283333  0.391667
    一般来说，X应该是一个二维数组，其中每行表示一个数据样本，每列表示一个特征。
    K-means算法会针对这些特征对所有数据点进行聚类，最终返回每个数据点所属的簇(cluster)标签，
    并生成相应的聚类中心。
    '''
    return centers


def plt_show():
    #todo 画图看下聚类效果

    centers = kemans()
    print(centers)

    plt.rcParams['font.size'] = 14
    colors = np.array(["red","green","blue","yellow"])

    #todo 散点图
    # 通过使用不同的颜色来表示不同的簇，即根据数据集中的cluster列的值将数据点分组
    plt.scatter(data['calories'],data['alcohol'],c=colors[data["cluster"]])


    #todo 在图表中标记出了每个簇的中心点，黑色十字表示中心点，其中横轴表示卡路里，纵轴表示酒精含量。

    plt.scatter(centers.calories,centers.alcohol,linewidths=3,marker='+',s=300,c='black')


    plt.xlabel("卡路里")
    plt.ylabel("酒精")

    plt.show()


def SSE():
    #TODO 计算SSE进行k值得选择
    clusters = 15
    K = range(1,clusters+1)
    TSSE = []
    std = StandardScaler()

    for k in K:

        SSE = []  #用于存储各个簇内差平方法和

        # 提取特征值 提取除了名字以外所有列信息
        X = data.loc[:, ['calories', 'sodium', 'alcohol', 'cost']]
        # print(X)
        #todo 增加数据标准化
        x = std.fit_transform(X)
        #使用聚类  默认使用的是 init='k-means++' 进行初始化
        #todo 首先，创建了一个KMeans对象kmean，并设置聚类中心的数量为3个。
        kmean = KMeans(n_clusters=k, init='k-means++')
        kmean.fit(x)

        #todo 返回簇类标签
        labels = kmean.labels_
        print("labels:",labels)
        #todo 返回簇类中心
        centers = kmean.cluster_centers_
        print("centers:", centers)
        #todo 计算各个簇的差平方和，存入列表中
        for label in set(labels):
            res = np.sum((x[labels == label,]-centers[label,:])**2)
            SSE.append(res)
        TSSE.append(sum(SSE))

    print(TSSE)
    print(len(TSSE))

    #todo 中文和负号的正常显示
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置绘图风格
    plt.style.use('ggplot')
    # 绘制K的个数与GSSE的关系
    plt.plot(K, TSSE, 'b*-')
    plt.xlabel('簇的个数')
    plt.ylabel('簇内离差平方和之和')
    # 显示图形
    plt.show()




#TODO 计算轮廓系数

def silhouette_():
    # 提取特征值 提取除了名字以外所有列信息
    X = data.loc[:, ['calories', 'sodium', 'alcohol', 'cost']]
    # print(X)

    #使用聚类  默认使用的是 init='k-means++' 进行初始化

    #todo 首先，创建了一个KMeans对象kmean，并设置聚类中心的数量为3个。
    kmean = KMeans(n_clusters=2, init='k-means++')


    #todo 增加数据标准化
    std = StandardScaler()
    x = std.fit_transform(X)


    #todo 然后，使用数据X对KMeans模型进行拟合（fit）操作，得到已经训练好的模型km。
    km = kmean.fit(x)


    #todo 将每个样本点所属的簇(cluster)标签（即聚类结果）添加到原始数据集data中，
    # 并存储在一个名为"cluster"的新列中。这一步操作涉及到km.labels_属性，可以得到每个数据点分配给哪个簇。
    data['cluster'] = km.labels_
    # print(data)
    '''
    km.labels_:是 KMeans 算法对每个样本点进行聚类后得到的簇标签。在 KMeans 模型中，每个簇都有一个编号（从0开始）
       name  calories  sodium  alcohol  cost  cluster
    0    燕京       144      15      4.5  0.43        1
    1    青岛       151      16      5.5  0.65        2
    2    崂山       157      17      4.5  0.56        2
    3    南阳       170      18      4.5  0.98        2
    4    郑州       152      19      4.6  0.54        2
    5    花旗       145      15      4.8  0.23        1
    6    乐视       125      16      4.6  0.65        0
    '''

    #todo 进行轮廓系数进行评估

    sil = silhouette_score(x,data.cluster)

    print(sil)



def sil_list():
    # todo 计算不同k值下的轮廓系数

    # 提取特征值 提取除了名字以外所有列信息
    X = data.loc[:, ['calories', 'sodium', 'alcohol', 'cost']]
    # print(X)

    #使用聚类  默认使用的是 init='k-means++' 进行初始化

    #todo 首先，创建了一个KMeans对象kmean，并设置聚类中心的数量为3个。
    kmean = KMeans(n_clusters=2, init='k-means++')


    #todo 增加数据标准化
    std = StandardScaler()
    x = std.fit_transform(X)


    #todo 然后，使用数据X对KMeans模型进行拟合（fit）操作，得到已经训练好的模型km。
    km = kmean.fit(x)


    #todo 将每个样本点所属的簇(cluster)标签（即聚类结果）添加到原始数据集data中，
    # 并存储在一个名为"cluster"的新列中。这一步操作涉及到km.labels_属性，可以得到每个数据点分配给哪个簇。
    data['cluster'] = km.labels_
    # print(data)
    '''
    km.labels_:是 KMeans 算法对每个样本点进行聚类后得到的簇标签。在 KMeans 模型中，每个簇都有一个编号（从0开始）
       name  calories  sodium  alcohol  cost  cluster
    0    燕京       144      15      4.5  0.43        1
    1    青岛       151      16      5.5  0.65        2
    2    崂山       157      17      4.5  0.56        2
    3    南阳       170      18      4.5  0.98        2
    4    郑州       152      19      4.6  0.54        2
    5    花旗       145      15      4.8  0.23        1
    6    乐视       125      16      4.6  0.65        0
    '''

    #todo 进行轮廓系数进行评估

    sil = silhouette_score(x,data.cluster)
    _score = []
    for i in range(2,16):
        labels = KMeans(n_clusters=i, init='k-means++').fit(x).labels_

        scores = silhouette_score(x,labels)

        _score.append(scores)


    plt.plot(list(range(2,16)),_score)

    plt.xlabel("簇类中心数目")
    plt.ylabel("轮廓系数")
    plt.show()

