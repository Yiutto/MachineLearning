# _*_ coding: utf-8 _*_
"""
    Created by Yiutto on 2019/8/8.
"""
import numpy as np
import operator
import os


def createDataSet():
    """
    构建训练数据集
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    """
    从文本中读取训练数据集
    :param filename:
    :return:
    """
    labelDict = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    arrayLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayLines)
    # 创建Numpy矩阵，构建0矩阵
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        # 将行数据用分隔符切分
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(labelDict.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    归一化特征值
    :param dataSet: np.array
    :return:
    """
    # 每列的最小值
    minVals = dataSet.min(0)
    # 每列的最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    # 行数
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 这里只是特征值相除
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals




def classify0(inX, dataSet, labels, k):
    """
    对未知类别属性的数据集中的每个点依次执行以下操作：
    1) 计算已知类别数据集中的点与当前点之间的距离；
    2) 按照距离递增次序排序；
    3) 选取与当前点距离最小的k个点；
    4) 确定前k个点所在类别的出现频率；
    5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
    :param inX: 输入向量（numpy.ndarray）
    :param dataSet: 输入的训练样本集（numpy.ndarray）
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return:
    """
    # 获取数组的行
    dataSetSize = dataSet.shape[0]

    # 将inX按照dataSetSize的行数复制，方便矩阵作运算
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 获取数组排序（增序）后的下标
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选出K个近邻数据的标签，建立字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    """
    验证结果，得到错误率
    :return:
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 数据的行数
    m = normMat.shape[0]
    # 测试集的个数
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # 简单的十字交叉验证
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                      datingLabels[numTestVecs:m], 3)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("The total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifyResult = classify0((inArr - minVals) / ranges, normDataSet, datingLabels, 3)
    print("You will probably like this person: " , resultList[classifyResult - 1])
def test1():
    group, labels = createDataSet()
    print(classify0([0, 0], group, labels, 3))
def test2():
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')

def img2vector(filename):
    """
    将一张图片32*32的文件转为一个向量[, 1024]
    :param filename:
    :return:
    """
    # 一张图片32*32
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    """
    手写数字识别系统的测试
    :return:
    """
    hwLabels = []
    # 获取目录内容
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("The classifier came back with: %d, the real answer is %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nThe total number of errors is: %d" % errorCount)
    print("\nThe total error rate is: %f" % (errorCount / float(mTest)))





if __name__ == '__main__':
    handwritingClassTest()
