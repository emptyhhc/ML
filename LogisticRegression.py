import math
import sys
import numpy as np
import time
import sklearn.datasets as skl_ds

def sigmoid(inX):
    try:
        ans = math.exp(-inX)
    except OverflowError:
        ans = float('inf')
    return 1.0/(1.0 + ans)

def gradDescent(data, label, weights, alpha = 0.01):

    res = np.dot(data, weights)
    yi = [[sigmoid(res[j]) for j in range(len(res))]]
    loss = [(label[x] - yi[0][x]) for x in range(len(yi[0]))]
    tidu = np.dot(data.transpose(), loss)
    print('bias = ', tidu[0])
    weights = weights + alpha * tidu
    return weights

def initFeat_data(batchSize, feat_num):
    feat_data = np.zeros((batchSize, feat_num))
    bias = np.ones((batchSize, 1))
    feat_data = np.c_[bias, feat_data]
    return feat_data

def train(data_url, weights, batchSize, feat_num):
    c = 0
    with open(data_url, 'r') as rfile:
        for x in range(5):
            rfile.seek(0)
            label = []
            feat_data = initFeat_data(batchSize, feat_num)
            count = 0
            for line in rfile:
                if(count == batchSize):
                    #weights = gradDescent(feat_data, label, weights)
                    alpha = 1.0 / (1.0 + np.sqrt(c))
                    weights = gradAscent1(weights, feat_data, label, alpha=alpha)
                    print(weights)
    #                print(c)
                    label = []
                    feat_data = initFeat_data(batchSize, feat_num)
                    count = 0
                    print('%d wan samples has trained.' % (c/10000))
                line = line.strip('\n')
                lineArr = line.split(' ')
                label.append(float(lineArr[0]))
                for k in range(1, len(lineArr)-1):
                    key = lineArr[k].split(':')[0]
                    value = lineArr[k].split(':')[1]
                    feat_data[count, (int(key)+1)] = float(value)
                count += 1
                c += 1
    return weights

def printModel(model_url, weights):
    with open(model_url, 'w') as wfile:
#        wfile.write('bias:')
        wfile.write(str(weights[0]))
        wfile.write('\n')
#        wfile.write('weigths: ')
        for i in range(1, len(weights)):
            wfile.write(str(i-1) + ':' + str(weights[i]))
            wfile.write(' ')

def sigmoid1(z):
    return 1 / (1 + np.exp(-z))

def gradAscent1(weights,dataMatIn,classLabels,alpha=0.001):
    dataMatrix=np.mat(dataMatIn) #(m,n)
    labelMat=np.mat(classLabels).transpose() #转置后(m,1)
    m,n=np.shape(dataMatrix)
    #weights=np.ones((n,1)) #初始化回归系数，(n,1)
     #定义步长

    h=sigmoid1(dataMatrix * np.mat(weights).transpose()) #sigmoid 函数
    error=labelMat - h #即y-h，（m,1）
    weights=weights - np.array(alpha * dataMatrix.transpose() * error).ravel()/len(dataMatrix) - weights * alpha * np.sign(weights) / len(dataMatrix)#梯度上升法

    return weights

def main():

    batchSize = 1000
    feat_num = 4000
    weights = np.random.rand(feat_num+1)
    #in_data = sys.argv[1]
    in_data = '0806'
    weights = train(in_data, weights, batchSize, feat_num)
    printModel('model.txt', weights)




if __name__ == '__main__':
    hour = time.localtime().tm_hour
    mins = time.localtime().tm_min
    main()
    times = time.localtime().tm_hour * 60 + time.localtime().tm_min - hour * 60 - mins
    print('finish all in %s' % str(time))





