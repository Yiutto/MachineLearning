## 决策树


### 优点
计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特
征数据。
### 缺点
可能会产生过度匹配问题。
### 适用数据范围
数值型和标称型
### 一般流程
- (1) 收集数据：可以使用任何方法。
- (2) 准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化。
- (3) 分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
- (4) 训练算法：构造树的数据结构。
- (5) 测试算法：使用经验树计算错误率。
- (6) 使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据
的内在含义。

### 部分代码
```angular2html
# sys.path.extend(['F:\\workspace\\python3\\MachineLearning\\decision_tree'])
import os
# 打印当前工作目录
print(os.getcwd()) 
# 改变当前目录
os.chdir('F:/workspace/python3/MachineLearning/decision_tree') 
import tree
myData, labels = tree.createDataSet()
myData
Out[9]: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
tree.calcShannonEnt(myData)
Out[10]: 0.9709505944546686
```
除了最大信息增益方法划分数据集，还有另一个度量集合无序程度的方法为基尼不纯度（Gini impurity)
```angular2html
import importlib
importlib.reload(tree)
tree.splitDataSet(myData, 0, 1)
Out[27]: [[1, 'yes'], [1, 'yes'], [0, 'no']]

tree.chooseBestFeatureToSplit(myData)
Out[29]: 0
```

递归创建决策树
```angularjs
myData, labels = tree.createDataSet()
myTree = tree.createTree(myData, labels)
```

通过plt构造决策树图
```angularjs
import importlib
importlib.reload(treePlotter)
tree1 = treePlotter.retrieveTree(1)
treePlotter.getNumLeafs(tree1)

# 加载数据
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = tree.createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)
```

### 小结
本章使用的算法为ID3，无法直接处理数值型数据（可以通过量化的方法将数值型数据转化为
标称数字），但是如果存在太多的特征划分，ID3算法仍然会有其他问题。

为了减少过度匹配问题，我们可以裁剪决策树，去掉一些不
必要的叶子节点。如果叶子节点只能增加少许信息，则可以删除该节点，将它并入到其他叶子节
点中，CART算法

