#Fixed increment perceptron learning algorithm for a classification problem with n input features(x1,x2,x3,...,xn) and two output classes 0 and 1

import numpy as np
from matplotlib import pyplot as plt

n = int(input('Enter the number of samples:'))
x = []
y = []

for i in range(n):
    temp = input()
    #print type(temp[0])
    #print type(temp[1])
    x.append(temp[0])
    y.append(temp[1])

x = np.array(x)
y = np.array(y)

#print x
#print y

w = np.arange(0,n)
#print w


net = x*w
#print net

def update_net(net,x,w,i):
    a = 0.003
    w[i] = w[i]-a*x[i]
    net = x*w
    for i in range(len(net)):
        if net[i] < 0:
            net[i] = 0
        else:
            net[i] = 1
    return net

for i in range(len(net)):
    if net[i] < 0:
        net[i] = 0
    else:
        net[i] = 1
print net

print type(net[0])
print type(y[0])

print (net[0]==y[0])

for i in range(n):
    count = 0
    while 1:
        if count<10000:
            break
        elif net[i] == y[i]:
            break
        else:
            net = update_net(net,x,w,i)
            count = count +1

print net
print y
