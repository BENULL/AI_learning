# coding = utf-8

from math import log
import numpy as np
from collections import Counter

class DecisionTree:
    """ID3 DecisionTree

    """

    def __init__(self):
        self.decisionTree = None
        self._X = None
        self._y = None

    # 计算信息熵
    def calcShannonEnt(self,y):
        lablesCounter = Counter(y)
        shannonEnt = 0.0
        for num in lablesCounter.values():
            p = num / len(y)
            shannonEnt += -p * log(p,2)
        return shannonEnt

    def fit(self, X, y):
        self._X = X
        self._y = y
        self.decisionTree = self.createTree(self._X,self._y)
        return self

    def splitDataset(self,X,y,d,value):
        features = X[X[:,d]==value]
        labels = y[X[:,d]==value]
        return np.concatenate((features[:,:d],features[:,d+1:]),axis=1), labels

    def chooseBestFeatureToSplit(self,X,y):
        numFeatures = X.shape[1]
        baseEntropy = self.calcShannonEnt(y)
        bestInfoGain, bestFeature = 0.0, -1

        for i in range(numFeatures):
            # 创建唯一的分类标签列表
            uniqueVals = np.unique(X[:,i])
            newEntropy =0.0
            # 计算每种划分方式的信息熵
            for value in uniqueVals:
                _x, _y = self.splitDataset(X,y,i,value)
                prob = len(_x)/len(X)
                newEntropy += prob * self.calcShannonEnt(_y)
            infoGain = baseEntropy - newEntropy
            if infoGain>bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
            return bestFeature

    def majorityCnt(self,y):
        lablesCounter = Counter(y)
        return lablesCounter.most_common(1)[0]

    def createTree(self,X,y):
        # 类别完全相同则停止继续划分
        if y[y == y[0]].size == y.size :
            return y[0]
        # 遍历完所有特征时返回出现次数最多的类别
        if X.shape[1] == 0:
            return self.majorityCnt(y)
        bestFeat = self.chooseBestFeatureToSplit(X,y)
        decisionTree = {bestFeat: {}}
        for value in np.unique(X[:,bestFeat]):
            decisionTree[bestFeat][value] = self.createTree(*self.splitDataset(X,y,bestFeat, value))
        return decisionTree

if __name__ == '__main__':

    dataSet = np.array([[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']])
    labels = ['no surfacing', 'flippers']
    dt = DecisionTree()
    X = dataSet[:, :2]
    X = X.astype(np.int)
    y = dataSet[:,-1]
    dt.fit(X,y)
    print(dt.decisionTree)
    from sklearn import datasets

    iris = datasets.load_iris()
    Z = iris.data[:,2:]
    M = iris.target
    dt2 = DecisionTree()
    dt2.fit(Z,M)
    print(dt2.decisionTree)