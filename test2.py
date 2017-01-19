#Implementation of backpropagation algorithm

import numpy as np

def normalise(x):
    xm = np.mean(x)
    xs = np.std(x)
    x = (x-xm)/xs
    x_max = max(abs(x))
    x = x/x_max
    return x,xm,(xs*x_max)

def denormalise(x,xm,xs):
    x = (x*xs) + xm
    return x

def train_data(w,v,input_var,del_v,del_w,output_var):
    count = 0
    while 1:
        input_hidden = np.matrix.transpose(v)*input_input
        output_hidden = 1/(1+np.exp(-input_hidden))
        input_output = np.transpose(w)*output_hidden
        output_output = 1/(1+np.exp(-input_output))
        error = (output_var - output_output) ** 2
        d = (output_var - output_output)*output_output*(1-output_output)
        y = output_hidden * d
        del_w = 0.8*del_w + 0.6*y
        e = w*d
        dd = e*np.matrix.transpose(output_hidden)*(1- output_hidden)
        x = input_input * np.matrix.transpose(dd)
        del_v = 0.8*del_v + 0.6*x
        v = v + del_v
        w = w + del_w
        er = np.array(error)
        erm = np.mean(er)
        count = count + 1
        if erm<0.001:
            break
        elif count<1000:
            break
    return w,v


i1 = np.matrix([0.4,0.3,0.6,0.2,0.1])
i2 = np.matrix([-0.7,-0.5,0.1,0.4,-0.2])
op = np.matrix([0.1,0.05,0.3,0.25,0.12])

w = np.random.rand(2,1)
v = np.random.rand(2,2)
del_w = np.matrix(np.zeros((2,1)))
del_v = np.matrix(np.zeros((2,2)))

w = np.matrix(w)
v = np.matrix(v)

i1, i1m , i1s = normalise(i1)
i2, i2m , i2s = normalise(i2)
op, opm , ops = normalise(op)

print i1
print i2
print op

for x in range(len(i1)):
    input_input = [i1[x],i2[x]]
    print input_input
    input_input = np.matrix(input_input)
    np.matrix([i1[x],i2[x]])
    output_var = np.matrix(op[x])
    w,v = train_data(w,v,del_v,del_w,input_var,output_var)
    print w
    print v

i1 = denormalise(i1,i1m,i1s)
i2 = denormalise(i2,i2m,i2s)
op = denormalise(op,opm,ops)

print i1
print i2
print op
