#! /usr/bin/python
# -*- coding:<UTF-8> -*-

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData = open(r'E:\tree.csv', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)
print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(featureList)

vec = DictVectorizer()

# 用函数把数据0/1化,并转成数组形式（必须转化）。
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:", str(dummyX))

# 分别显示出每类属性的每个值
print(vec.get_feature_names())

# 原始数据最终的分类标签
print("labelList:" + str(labelList))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY" + str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf:" + str(clf))

with open(allElectronicsData.doc) as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
oneRowX = dummyX[0, :]
print("oneRowX" + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX:" + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY:" + str(predictedY))
