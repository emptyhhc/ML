import sys
import math
import time

def sigmoid(inX):
    try:
        ans = math.exp(-inX)
    except OverflowError:
        ans = float('inf')
    return 1.0/(1 + ans)


def loadWeights(file_url):
    bias = 0
    weights = []
    with open(file_url,'r') as file:
        lines = file.readlines()
        bias = float(lines[0].strip('\n'))
        w = lines[1].strip('\n').split(' ')
        for x in range(1, len(w)-1):
            kv = w[x]
            key = kv.split(':')[0]
            value = kv.split(':')[1]
            weights.append(float(value))
    return  bias, weights

def predict(bias, weights):
    #input_url = sys.argv[1]
    input_url = '0806'
    label = []
    predictRes = []
    with open(input_url, 'r') as file:
        for line in file:
            line = line.strip('\n')
            lineArr = line.split(' ')
            label.append(lineArr[0])
            sum = bias
            for x in range(1, len(lineArr)-1):
                kv = lineArr[x].split(':')
                key = kv[0]
                value = kv[1]
                sum += weights[int(key)] * float(value)
                predictRes.append(sigmoid(sum))
    return label, predictRes

def printResult(output_url, label, predict):
    with open(output_url, 'w') as wfile:
        for x in range(len(label)):
            wfile.write(str(label[x]))
            wfile.write(' ')
            wfile.write(str(predict[x]))
            wfile.write('\n')

def main():
    weights_url = 'model.txt'
    bias, weights = loadWeights(weights_url)
    outputResult = 'predict.txt'
    real, predictRes = predict(bias, weights)
    printResult(outputResult, real, predictRes)

if __name__ == '__main__':
    hour = time.localtime().tm_hour
    mins = time.localtime().tm_min
    main()
    times = time.localtime().tm_hour * 60 + time.localtime().tm_min - hour * 60 - mins
    print('finish all in %s' % str(time))
