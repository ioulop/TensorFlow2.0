#!/usr/bin/env python
# -*-coding:<UTF-8> -*-
from sklearn import datasets
import csv
import random
import math
import operator


def loadDataset(filename, split, trainingSet=[], testData=[]):
    # with open(filename) as csvfile:
        lines = csv.reader(filename)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = dataset[x][y]
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testData.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += math.pow((instance1[x], instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x].length)
        distances.append(trainingSet[x], dist)
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1))
    return sortedVotes[0][0]

def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


if __name__ == '__main__':
    trainingSet = []
    testSet = []
    split = 0.67
    iris = datasets.load_iris()
    print(iris)

    loadDataset(iris, split, trainingSet, testSet)
    print("trainingSet:" + repr(len(trainingSet)))
    print("testSet:" + repr(len(testSet)))
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:' + repr(accuracy) + '%')