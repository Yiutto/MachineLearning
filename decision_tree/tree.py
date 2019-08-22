# _*_ coding: utf-8 _*_
"""
    Created by Yiutto on 2019/8/10.
"""

from math import log
import operator
import os

def calcShannonEnt(dataSet):
    """
    计算香农嫡
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底求对数
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    """
    创建测试数据集，计算熵
    :return:
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # 特征属性
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 特征值
    :return:
    """
    # 创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 将value剔除
            reducedFeatVec = featVec[:axis]
            # 注意下面的extend，与append是完全不同的
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet: 列表组成的元素的列表
    :return:最好特征的索引值
    """
    # 这里面数据的最后一列为当前实例的类别标签
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfogain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        # 得到当前特征中的所有唯一属性值
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵（对每个特征）
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 对所有唯一特征值得到的熵求和
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息增益
        if (infoGain > bestInfogain):
            bestInfogain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    返回出现次数最多的分类名称
    :param classList:
    :return:
    """
    # 创建数据字典
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 排序 (按value，倒序)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
     递归构建决策树，生成规则
    :param dataSet: 数据集
    :param labels: 特征属性标签列表
    :return:
    """
    # 获取类别名
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分（递归函数的第一个停止条件，所有类标签完全相同）
    if classList.count(classList[0]) == len(classList):
        # 直接返回改类标签
        return classList[0]
    # 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    "1.选择最好的特征"
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 构建数据类型存储树
    myTree = {bestFeatLabel: {}}
    # 删除labels中bestFeat索引对应的值
    del(labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    "2.根据最好特征，通过值来划分相关数据集"
    # 遍历当前选择特征包含的所有属性值
    for value in uniqueVals:
        # 复制类标签
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转化为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: # 到达叶子节点，则返回当前节点的分类标签
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    """
    保存决策树的模型
    :param inputTree:
    :param filename:
    :return:
    """
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    """
    读取pick模型
    :param filename:
    :return:
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)

if __name__ == '__main__':
    myData, labels = createDataSet()
    createTree(myData, labels)