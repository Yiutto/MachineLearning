## logistic回归
利用Logistic回归进行分类的主要思想是：根据现有数据对分类边界线建立回归公式，以此进行分类。
### 优点
计算代价不高，易于理解和实现
### 缺点
容易欠拟合，分类精度可能不高
### 适用数据范围
标称型和数值型
### 一般流程
- (1) 收集数据：采用任意方法收集数据。
- (2) 准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据
格式则最佳。
- (3) 分析数据：采用任意方法对数据进行分析。
- (4) 训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
- (5) 测试算法：一旦训练步骤完成，分类将会很快。
- (6) 使用算法：首先，我们需要输入一些数据，并将其转换成对应的结构化数值；
接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于
哪个类别；在这之后，我们就可以在输出的类别上做一些其他分析工作。

梯度上升法

要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻。

你最经常听到的应该是梯度下降算法，它与这里的梯度上升算法是一样的，只是公式中的
加法需要变成减法。
梯度上升算法用来求函数的最大值，而梯度下降算法用来求函数的最小值。


### 实现方式
随机梯度上升

梯度上升算法在每次更新回归系数时都需要遍历整个数据集，该方法在处理100个左右的数
据集时尚可，但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度就太高了。一种
改进方法是一次仅用一个样本点来更新回归系数，该方法称为随机梯度上升算法。由于可以在新
样本到来时对分类器进行增量式更新，因而随机梯度上升算法是一个在线学习算法。与“在线学
习”相对应，一次处理所有数据被称作是“批处理”。


### 部分代码
```angular2html
# sys.path.extend(['F:\\workspace\\python3\\MachineLearning\\logistic'])
import os
# 打印当前工作目录
print(os.getcwd()) 
# 改变当前目录
os.chdir('F:/workspace/python3/MachineLearning/logistic') 
import logRegres

import importlib
importlib.reload(logRegres)
Out[53]: <module 'logistic.logRegres' from 'F:\\workspace\\python3\\MachineLearning\\logistic\\logRegres.py'>
dataArr, labelMat = logistic.logRegres.loadDataSet()
weights = logistic.logRegres.gradAscent(dataArr, labelMat)

# 画逻辑回归分割图
logistic.logRegres.plotBestFit(weights)

# 随机梯度上升
weights = logistic.logRegres.stocGradAscent0(array(dataArr), labelMat)
logistic.logRegres.plotBestFit(weights)

# 小批量随机梯度上升
dataArr, labelMat = logRegres.loadDataSet()
weights = logRegres.stocGradAscent1(array(dataArr), labelMat, 500)
logRegres.plotBestFit(weights)

```
#### 伪代码
```angularjs
梯度上升法的伪代码如下：
每个回归系数初始化为1 
重复R次：
    计算整个数据集的梯度
    使用alpha × gradient更新回归系数的向量
    返回回归系数  
```
```angularjs
随机梯度上升算法可以写成如下的伪代码：
所有回归系数初始化为1 
对数据集中每个样本
    计算该样本的梯度
    使用alpha × gradient更新回归系数值
返回回归系数值   
```
#### 对缺失的数据处理
 使用可用特征的均值来填补缺失值；
 使用特殊值来填补缺失值，如1；  忽略有缺失值的样本；
 使用相似样本的均值添补缺失值；
 使用另外的机器学习算法预测缺失值。


#### 示例：使用Logistic回归估计马疝病的死亡率
(1) 收集数据：给定数据文件。
(2) 准备数据：用Python解析文本文件并填充缺失值。
(3) 分析数据：可视化并观察数据。
(4) 训练算法：使用优化算法，找到最佳的系数。
(5) 测试算法：为了量化回归的效果，需要观察错误率。根据错误率决定是否回退到训练
阶段，通过改变迭代的次数和步长等参数来得到更好的回归系数。
(6) 使用算法：实现一个简单的命令行程序来收集马的症状并输出预测结果并非难事，这
可以做为留给读者的一道习题。

```angularjs
import importlib
importlib.reload(logRegres)
logRegres.multiTest()
```

### 本章小结
Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化
算法来完成。在最优化算法中，最常用的就是梯度上升算法，而梯度上升算法又可以简化为随机
梯度上升算法。
随机梯度上升算法与梯度上升算法的效果相当，但占用更少的计算资源。此外，随机梯度上
升是一个在线算法，它可以在新数据到来时就完成参数更新，而不需要重新读取整个数据集来进
行批处理运算。