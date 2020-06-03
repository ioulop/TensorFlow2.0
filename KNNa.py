from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
import numpy as np

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    '''
    处理初始数据：文件转矩阵，把收集到的数据进行矩阵化，好方便处理
    :param filename: 文本文件的名字
    :return: 返回两个矩阵，一个数据的矩阵，一个标签的矩阵
    '''

    f = open(filename)
    line = f.readline()
    data_array = []
    data1 = []
    datalabel = []
    while line:
        line = line.strip('\n')
        num = list(map(str, line.split(',')))
        # line.split(',')表示把文件的每一行以逗号‘，’分隔开，分成多个字符串存在一个列表里
        # map(str,line.split(',')) 表示把分开后的字符串转换成需要的类型，这里的str可以换成int，float等，并返回迭代器。这里的map函数是一个迭代器
        # 整体表示用list函数把map函数返回的迭代器遍历展开成一个列表
        data_array.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_array)
    a = data_array.shape[0]  # 行数
    b = data_array.shape[1]  # 列数
    index = 0
    returnMat = zeros((a, 3))  # 创建了一个和原文件数据同行数，同列数的零矩阵
    for i in range(a):
        for j in range(b-1):
            data1.append(float(data_array[i][j]))
        returnMat[index, :] = data1[0:3]  # 把该行数据放入到矩阵中
        index += 1
        data1 = []  # 放入之后把该行数据清零，以便下一行数据可以顺利放进去
        datalabel.append(int(data_array[i][-1]))  # 把最后的标签数据放入标签列表中
    print(datalabel)
    return returnMat, datalabel


def makeMatplotgraph(file):
    '''
    分析数据：该函数使用 Matplotlib 创建散点图
    :param file: 输入要画图的数据文件
    :return: 散点图

    '''

    datingDataMat, label = file2matrix(file)
    fig = plt.figure()  # 新建画布
    ax = fig.add_subplot(111)  # 将画布分成一行一列图放在第一个位置
    print(datingDataMat)
    print(type(array(label)))
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 150*array(label))
    plt.show()
    plt.close()


def autoNorm(dataSet):
    '''
    归一化特征值
    :param dataset: 数据集
    :return:
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
