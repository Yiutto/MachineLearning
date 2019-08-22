# _*_ coding: utf-8 _*_
"""
    Created by Yiutto on 2019/8/22.
"""
from numpy import *
def loadDataSet():
    """
    创建实验样本
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 代表侮辱性文字, 0 代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    """
    根据训练集生成词汇表
    :param dataSet:
    :return:
    """
    # 新建一个空集
    vocabSet = set()
    for document in dataSet:
        # 创建2个集合的并集（按位或操作符）
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    根据词汇表建立文档矩阵
    :param vocabList: 词汇表
    :param inputSet: 输入文档
    :return:
    """
    # 创建一个所有元素都为0的向量
    returnVec = len(vocabList) * [0]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    """
    根据词汇表建立文档矩阵
    :param vocabList: 词汇表
    :param inputSet: 输入文档
    :return:
    """
    # 创建一个所有元素都为0的向量
    returnVec = len(vocabList) * [0]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    """
    得到训练文档各个词的条件概率和类标签概率
    :param trainMatrix: 训练文档（arr）
    :param trainCategory: 类标签（arr）
    :return:
    """
    # 训练文档的数目
    numTrainDocs = len(trainMatrix)
    # 训练文档的词个数
    numWords = len(trainMatrix[0])
    # 标签1出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化各个类别下词的出现的次数
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    # 初始化各个类别下所有词的总数
    # p0Denom = 0.0; p1Denom = 0.0
    # 为了防止其中一个概率值为0，那么最后的乘积也为0。
    p0Num = ones(numWords);p1Num = ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            # 标签=1中各个词出现次数的统计
            p1Num += trainMatrix[i]
            # 标签=1中所有词出现次数的统计
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 求出各个类下每个词的概率
    # p1Vect = p1Num / p1Denom # change to log()
    # p0Vect = p0Num / p0Denom
    # 防止数据下溢出，这里求对数
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)
    # 在代数中有ln(a * b) = ln(a) + ln(b)，
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if (p1 > p0):
        return 1
    else:
        return 0

def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    # 定义文档矩阵
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry1 = ['love', 'my', 'dalmation']
    thisDoc1 = setOfWords2Vec(myVocabList, testEntry1)
    print(testEntry1, 'classified as: ' , classifyNB(thisDoc1, p0V, p1V, pAb))
    testEntry2 = ['stupid', 'garbage']
    thisDoc2 = setOfWords2Vec(myVocabList, testEntry2)
    print(testEntry2, 'classified as: ' , classifyNB(thisDoc2, p0V, p1V, pAb))

def textParse(bigString):
    """
    解析字符串
    :param bigString:
    :return:
    """
    import re
    # 利用正则表达式提取单词
    listOfTokens = re.split(r'\W*', bigString)
    # 过滤掉一些长度小于2的字符（空字符等）
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):


if __name__ == '__main__':
    testingNB()


