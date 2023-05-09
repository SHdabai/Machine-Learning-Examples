# -*- coding: UTF-8 -*-
from pprint import pprint
import numpy as np
from functools import reduce

#todo 加载数据查看数据格式  使用jieba分词可以达到这个效果
def loadData():
    postingList=[
        ['今天', '我', '很','开心'],                #切分的词条
        ['考试', '100分', '心情','高兴'],            #切分的词条
        ['工作', '加薪', '很','兴奋'],                #切分的词条
        ['狗子', '跑', '了','伤心'],                #切分的词条
        ['头痛', '心里', '很','痛苦'],                #切分的词条
        ]
    classVec = [0,0,0,1,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

#todo 构造词典，因为我们使用的词袋模型
def createVocabList(data):
    '''
    :param data: 就是我们的样本数据集，目的就是构造
    包含所有词的词典
    :return:  是一个大的表，包含样本中的所有词
    '''
    voc = set([]) #这里要加上set要不下边取并集时候会报错
    for doc in data:
        voc = voc | set(doc) #取并集，保证每个词的ID唯一
    return list(voc)

#todo 将我们的样本数据进行向量化处理
def wordVec(VocabList,data):
    '''
    :param VocabList: 这个是所有词的构成词典
    :param data: 经过数据处理的原始语料数据,注意这里是单条数据
    :return: 样本数据向量化结果
    '''
    vocList = [0] * len(VocabList)   #创建向量词典列表。初始化为0
    for _ in data: #遍历整个语料信息
        #todo 进行判断，样本数据是否在词表中存在，存在计为1
        if _ in VocabList:
            vocList[VocabList.index(_)] = 1

        else:
            # print(f"{_}不存在")
            continue

    return vocList
'''
这里说明一下，这也是词袋模型的缺点所在。当我们遇到测试用例的数据
在词表中不存在的时候，我们的模型就抓瞎了。因此这里可以对词表进行再
丰富，或者使用其他的向量化手段；例如：NLP中的Word2vector手段
这里仅仅是个学习展示。
'''

#TODO  开始构造贝叶斯分类器
'''
还记得我们第一章，中我们推导的朴素贝叶斯公式，想要求得：后验概率P(A|B)
我们要求得：
1. 先验概率P(A)
2. 条件概率P(B|A)
3. 至于P(B)这个在第一章我们知道这个对模型最终结果影响不大，不必处理，因此
重点求得：1，2

我们的模型最终给我返回三个东西：
1. vec_0 --- 开心的条件概率组
2. vec_1 --- 伤心的条件概率组
3. simple_1 --- 样本数据属于伤心的概率

'''
def train(data,label):
    '''
    :param data: 这里的数据是样本经过向量化后的向量信息
    :param label:  样本所对应的标签信息
    :return:
    1. vec_0 --- 开心的条件概率组
    2. vec_1 --- 伤心的条件概率组
    3. simple_1 --- 样本数据属于伤心的概率
    '''
    num_simple = len(data) #样本的数量
    num_words = len(data[0]) #统计每个样本的词的数据量

    simple_1 = sum(label)/float(num_simple)  #计算的是伤心的
    '''
    这里说明一下，为什么使用的是sum(label) 来进行计算：
    正常这里应该是，属于伤心的样本数据/总的样本数
    那么我们这里用sum的效果就是，伤心样本的数量，因为伤心
    的label为1，开心为0，所有label的和就是，伤心的数量
    这里取巧了。
    '''
    # 进行条件概率数组初始化
    array_0,array_1 = np.zeros(num_words),np.zeros(num_words)
    _0 = 0.0
    _1 = 0.0 #这里是多求条件概率的分母进行初始化，至于怎么用后面说

    #todo 对所有伤心的样本进行计算条件概率
    for i in range(num_simple):
        if label[i] == 1: #伤心
            array_1 += data[i]
            _1 += sum(data[i])
        else: #开心
            array_0 += data[i]
            _0 += sum(data[i])

    vec_0 =  array_0/_0
    vec_1 =  array_1/_1
    return vec_0,vec_1,simple_1


# todo 套用计算公式
def prod_fun(vec_test,vec_0,vec_1,simple_1):#进行赋值计算
    '''
    :param vec_test: 测试样本的样本向量
    :param vec_0: 开心的条件概率
    :param vec_1: 伤心的条件概率
    :param simple_1: 伤心的先验概率
    :return: label信息
    '''
    #todo 参考  p(C|X) = p(C)p(X|C) simple_1 就是P(C)即先验概率。
    pro_0 = reduce(lambda x,y:x*y,vec_test*vec_0) * (1 - simple_1)
    pro_1 = reduce(lambda x,y:x*y,vec_test*vec_1) * simple_1
    if pro_0 > pro_1:
        return 0
    else:
        return 1

#todo  对测试样例进行向量化并且进行使用
def test():
    postingList, classVec = loadData()
    #词表
    my_voc = createVocabList(postingList)
    #训练数据
    train_data = []
    for i in postingList:
        train_data.append(wordVec(my_voc,i))
    vec_0,vec_1,simple_1 = train(train_data,classVec)

    test_01 = ["中奖","了","开心"]
    #todo 开始进行测试
    test_simple_01 = np.array(wordVec(my_voc,test_01))
    res = prod_fun(test_simple_01, vec_0, vec_1, simple_1)
    if prod_fun(test_simple_01,vec_0,vec_1,simple_1):
        print(f"{test_01}预测标签为:开心{res}")
    else:
        print(f"{test_01}预测标签为:伤心{res}")

    test_02 = ["表白","失败","很","痛苦"]

    #todo 开始进行测试
    test_simple_02 = np.array(wordVec(my_voc,test_02))
    res = prod_fun(test_simple_02, vec_0, vec_1, simple_1)
    if prod_fun(test_simple_02,vec_0,vec_1,simple_1):

        print(f"{test_02}预测标签为:开心{res}")
    else:
        print(f"{test_02}预测标签为:伤心{res}")


if __name__ == '__main__':
    postingList, classVec = loadData()
    # for each in postingList:
    #     print(each)
    pprint(postingList)
    print(classVec)
    #词表
    my_voc = createVocabList(postingList)
    print(f"my_voc:{my_voc}")
    #训练数据
    train_data = []
    for i in postingList:
        train_data.append(wordVec(my_voc,i))

    pprint(f"train_data:{train_data}")
    vec_0,vec_1,simple_1 = train(train_data,classVec)
    pprint(f"vec_0:\n{vec_0}")
    pprint(f"vec_1：\n{vec_1}")
    print(f"classVec:{classVec}")
    print(f"simple_1：{simple_1}")

    print("=====================================================")

    test()






















