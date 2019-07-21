import numpy as np
import operator
import os
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()


def classify0(inx, dataset, labels, k):
    dataSetSize = dataset.shape[0]#得到dataset的维度
    diffmat = np.tile(inx, (dataSetSize, 1)) - dataset#复制inx，让它与dataset里的每一个元素相减
    sqDiffmat = diffmat ** 2
    sqDistance = sqDiffmat.sum(axis=1)
    distances = sqDistance ** 0.5#计算欧式距离
    sortedDistIndicies = distances.argsort()#排序，由小到大，返回元素下标
    classCount = {}#创建一个字典
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]#按照距离由小到大，把labels排序，取出前k个
        classCount[votelabel] = classCount.get(votelabel, 0) + 1#统计label出现次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#降序排列，按照第二列元素排
    return sortedClassCount[0][0]#返回出现最高的类别当做当前分类

#把文件转换成矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOFLines = fr.readlines()#按行读取，包括\n
    numberOFLines = len(arrayOFLines)#获取长度
    returnMat = np.zeros((numberOFLines, 3))#创建一个1000*3的矩阵
    classLabelVector = []
    index = 0
    label = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    for line in arrayOFLines:
        line = line.strip()#删除换行符
        listFromLine = line.split('\t')#按制表符切开
        returnMat[index, :] = listFromLine[0:3]#把每行前三列放到刚才创建的matrix中
        #print(returnMat)
        classLabelVector.append(label[listFromLine[-1]])#把标签提取出来，单独形成一个矩阵，与特征矩阵对应
        index += 1
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix(r"E:\Users\Administrator\PycharmProjects\untitled9\datingTestSet.txt")

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title(u'散点图')
plt.xlabel(u'打机时间')
plt.ylabel(u'飞机里程')
#plt.show()
'''

#归一化 公式=（oldvalue-min)/(max-min)
def autoNorm(dataset):
    minVals = dataset.min(0)#找到最小值
    maxVals = dataset.max(0)#最大值
    ranges = maxVals - minVals
    normDataset = np.zeros(np.shape(dataset))#创建一个维度与dataset一样的0矩阵
    m = dataset.shape[0]#得到dataset的维度
    normDataset = dataset - np.tile(minVals, (m, 1))#oldvalue-min
    normDataset = normDataset / np.tile(ranges, (m, 1))#(oldvalue-min)/(max-min)
    return normDataset, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(r"E:\Users\Administrator\PycharmProjects\untitled9\datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorcount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        #[i,:]代表第i行所有元素
        print("the classifier came back with：%d , the real answer is :%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorcount += 1.0
    print("the total error rate is : %f" % (errorcount / float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))#input可以手动输入
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix(r"E:\Users\Administrator\PycharmProjects\untitled9\datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])#令输入的数据生成一个numpy矩阵
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)#对新生成的矩阵归一化之后，放入分类器
    print('You probably like this person:%s' % (resultList[(classifierResult) - 1]))

#处理图像
#fr = open(r"E:\Users\Administrator\PycharmProjects\untitled9\testDigits\0_13.txt")
#f = fr.read().splitlines()
#print(type(f))

#图像转换为向量，把0，1存进numpy数组里
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#testvector = img2vector(r"E:\Users\Administrator\PycharmProjects\untitled9\testDigits\0_13.txt")
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir(r"E:\Users\Administrator\PycharmProjects\untitled9\trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split(',')[0]#按’,'切割，获取文件名
        #print(fileStr)
        classNumStr = int(fileStr.split('_')[0])#以'_'为分隔符，获取文件名第一个字符，第一个字符就代表什么数字
        #print(classNumStr)
        hwLabels.append(classNumStr)#把标签存起来，也就是文件名的第一个字符
        trainingMat[i,:] = img2vector(r'E:\Users\Administrator\PycharmProjects\untitled9\trainingDigits/%s'%(filenameStr))
        #把每个文件都转成numpy数组
    testFilelist = os.listdir(r"E:\Users\Administrator\PycharmProjects\untitled9\testDigits")
    errorCount = 0.0
    mTest = len(testFilelist)
    for i in range(mTest):
        filenameStr = testFilelist[i]
        fileStr = filenameStr.split(',')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'E:\Users\Administrator\PycharmProjects\untitled9\testDigits/%s' % (filenameStr))
        classifyResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        #print('the classifier came back with :%d,the real answer is :%d'%(classifyResult,classNumStr))
        if(classifyResult != classNumStr): errorCount += 1.0
    #print('\n the total number of errors is :%d'%(errorCount))
    #print('\n the total error rate is :%f'%(errorCount/float(mTest)))

handwritingClassTest()
