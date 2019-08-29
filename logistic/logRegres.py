# _*_ coding: utf-8 _*_
"""
    Created by Yiutto on 2019/8/23.
"""
import math
from numpy import *
def loadDataSet():
    """
    加载数据集
    :return:
    """
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    # return 1.0 / (1 + math.exp(-inX))
    # 这里要区别math里面的exp和numpy里面的exp
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    用梯度下降法求得weight
    :param dataMatIn: 100x3
    :param classLabels: 1x100行向量
    :return:
    """
    # 转化为NumPy矩阵数据类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    # 学习率
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.A1

def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升算法
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升算法（小批量）
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha每次迭代时需要调整
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    """
    画图，拟合一条逻辑回归分界线
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    # 注意numpy.ndarray和 numpy.matrixlib.defmatrix.matrix区别
    dataArr = array(dataMat)
    # dataArr的行数
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # WX = 0 (w0 + w1X + w2Y = 0)
    y = (- weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    """
    小批量随机梯度算法进行分类
    :return:
    """
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

