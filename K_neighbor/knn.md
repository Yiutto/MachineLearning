## k-近邻算法
存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据
与所属分类的对应关系。输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的
特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。一般来说，我们
只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。
最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。
### 优点
精度高、对异常值不敏感、无数据输入假定。
### 缺点
计算复杂度高、空间复杂度高。
### 适用数据范围
数值型和标称型
### 一般流程
- (1) 收集数据：可以使用任何方法。
- (2) 准备数据：距离计算所需要的数值，最好是结构化的数据格式。
- (3) 分析数据：可以使用任何方法。
- (4) 训练算法：此步骤不适用于k-近邻算法。
- (5) 测试算法：计算错误率。
- (6) 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输
入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。

### 部分代码
```angular2html
# sys.path.extend(['F:\\workspace\\python3\\MachineLearning\\K_neighbor'])
import os
# 打印当前工作目录
print(os.getcwd()) 
# 改变当前目录
os.chdir('F:/workspace/python3/MachineLearning/K_neighbor') 
import KNN
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet.txt')

# 画散点图
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
plt.show()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
plt.show()

testVector = KNN.img2vector('testDigits/0_13.txt')
testVector
Out[7]: array([[0., 0., 0., ..., 0., 0., 0.]])
testVector[0:31]
Out[8]: array([[0., 0., 0., ..., 0., 0., 0.]])

```

#### 数值归一化
newValue = (oldValue - minValue) / (maxValue - minValue)

### 小结
K决策树是K-近邻算法的优化班，可以节省大量的计算开销.

K-近邻算法必须保存全部数据集，如果训练数据集很大，必须使用大量的存储空间。此外，
由于必须对数据集中的每个数据计算距离值，实际使用时可能非常耗时。
K-近邻算法的另一个缺陷是它无法给出任何数据的基础结构信息，因此我们也无法知晓平均
实例样本喝典型实例样本具有什么特征。（概率测量方法处理分类问题可以解决这个问题）