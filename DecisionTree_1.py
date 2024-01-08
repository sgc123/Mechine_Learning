from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

def createDataSet():
	dataSet = [[0, 0, 0, 0, 'no'],[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']		
	return dataSet, labels

def createTree(dataset,labels,featLabels):  
	classList = [example[-1] for example in dataset]  #dataset最后一列为标签
    
    #两个递归停止条件
	if classList.count(classList[0]) == len(classList):  #判断是否属于一类
		return classList[0]  #如果是一类，那么停止递归
	if len(dataset[0]) == 1:  #当前特征只剩下一类标签：特征都做完了，那么返回大多数列
		return majorityCnt(classList)    

    
	bestFeat = chooseBestFeatureToSplit(dataset)  #选取最优的特征的索引值
	bestFeatLabel = labels[bestFeat]  #找到特征的名字
	featLabels.append(bestFeatLabel)  #featLabels添加特征名称
	myTree = {bestFeatLabel:{}}  #字典结构嵌套字典，设定根节点

	del labels[bestFeat] #选取完成特征后，将dataset 中的 labels中的特征进行删除

	featValue = [example[bestFeat] for example in dataset]  #从dataset中拿出具体的列
	uniqueVals = set(featValue)  #统计当前列属性值有几个
	for value in uniqueVals:#进行当前属性值不同样式的for循环
		sublabels = labels[:]  #拿出labels
		#递归调用
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset,bestFeat,value),sublabels,featLabels)  
	return myTree

def majorityCnt(classList):  #传入参数 分类列
	classCount={}  #计数
	for vote in classList: #分类列
		if vote not in classCount.keys():classCount[vote] = 0  #如果不在当前类别，重新创建一个类别
		classCount[vote] += 1
	sortedclassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedclassCount[0][0]  #返回出现次数最多的 假设贷款 是否贷款给这个人

def chooseBestFeatureToSplit(dataset):  #选择最好的特征，主要函数
	numFeatures = len(dataset[0]) - 1  #特征的数量
	baseEntropy = calcShannonEnt(dataset)  #计算熵值
	bestInfoGain = 0  #信息增益
	bestFeature = -1  #最好的特征
	for i in range(numFeatures):  #循环样本的个数
		featList = [example[i] for example in dataset]  #拿到当前的特征（取这一列）
		uniqueVals = set(featList) #该特征有多少不同的值
		newEntropy = 0  #
		for val in uniqueVals:  #对于每一个特征的不同的值进行遍历
			subDataSet = splitDataSet(dataset,i,val)
			prob = len(subDataSet)/float(len(dataset))  #当前比例
			newEntropy += prob * calcShannonEnt(subDataSet)  #计算新的熵值
		infoGain = baseEntropy - newEntropy  #信息增益
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain  #查看最佳的信息增益
			bestFeature = i	  #最好的特征
	return bestFeature
				
		
def splitDataSet(dataset,axis,val):  #传入数据集，哪个特征，特征值
	retDataSet = []
	for featVec in dataset:
		if featVec[axis] == val:  #查找特征=val值的一列
			reducedFeatVec = featVec[:axis]  #去掉这一列 
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet  #返回不带这一列的数据集
			
def calcShannonEnt(dataset):
	numexamples = len(dataset)  #样本的个数
	labelCounts = {}  #对于每一个标签出现的个数 ，yes出现多少个 no 出现多少个
	for featVec in dataset:   #对每个样本进行遍历
		currentlabel = featVec[-1]  #查看是否贷款
		if currentlabel not in labelCounts.keys(): #如果该类别不存在
			labelCounts[currentlabel] = 0
		labelCounts[currentlabel] += 1  #数量加一
		
	shannonEnt = 0  #熵值初始化
	for key in labelCounts:  #对每一个类别进行遍历 
		prop = float(labelCounts[key])/numexamples #计算该类别的概率
		shannonEnt -= prop*log(prop,2)  #计算该类别的熵值
	return shannonEnt  


if __name__ == '__main__':
	dataset, labels = createDataSet()
	featLabels = []
	myTree = createTree(dataset,labels,featLabels)

