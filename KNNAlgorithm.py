#! /usr/bin/env python
# -*- coding:<uft-8> -*-
import numpy
from numpy import *
import KNNa
import operator

# b = random.random(4)  # random 生成一个列表，列表里面有四个元素
# a = random.rand(4, 4)  # rand  生成一个二维矩阵，括号里的参数分别代表行和列的元素个数
# randMat = mat(b)  # mat函数，把数组转化为矩阵
# print(b)
# print(randMat.I)  # .I  求出一个矩阵的逆矩阵
# print(a + b) #矩阵做加法：二维矩阵的每一行都加上列表里面的值
group, labels = KNNa.file2matrix(r'E:\file.txt')
# print(group)
# print(labels)
normDataSet, ranges, minVal = KNNa.autoNorm(group)
print(normDataSet)
print(ranges)
print(minVal)

'''
    K近邻：使用欧式距离的K近邻算法
    :param inx:未知的数据(集)，即待分类的数据集 
    :param dataSet: 测试数据，即已知分类的数据，不包含其最后的分类标签
    :param labels:  测试数据最后的结果标签
    :param k: 最近邻的个数
    :return:返回最后未知数据的分类标签 
'''
def classify0(inx, dataSet, labels, k):

    # shape[0] 返回矩阵第一维度的长度
    numpy.array(dataSet)
    dataSetSize = numpy.array(dataSet).shape[0]
    # 按行复制，将输入向量扩展到和dataSet一样的行数，
    diffMat = tile(inx, (dataSetSize, 1))
    diffMat = diffMat - dataSet  # 再减去dataSet
    sqDiffMat = diffMat ** 2
    # 按行相加求和,二维数组中，axis = 1 代表按行相加，axis = 0 代表按列相加，一维数组的axis=0，按列相加，因为他就一行所以没必要写
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # argsort()函数，是将distances 的元素按照从小到大的顺序进行排序，并把各个值的index返回
    sortedDistances = distances.argsort()
    # 以上就是求输入向量和各个训练样本之间的欧式距离
    # argsort()排序后返回下标向量

    # 初始化一个字典，也就是C++中的map
    classCount = {}
    for i in range(k):
        # 通过循环，依次找出前k距离的下标（第几行样本，也就是第几个样本),映射到labels上，找出对应分类label
        voteIlabel = labels[sortedDistances[i]]
        # 字典：{'标签'：出现次数}
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# new = classify0([9000, 86820, 789], group, labels, 3)
# print(new)



