import numpy as np

def train_data(w,v,input_var,output_var):
    count = 0
    del_w = np.matrix(np.zeros((3,1)))
    #print del_w
    del_v = np.matrix(np.zeros((2,3)))
    while 1:
        input_hidden = np.matrix.transpose(v)*input_input
        output_hidden = 1/(1+np.exp(-input_hidden))
        input_output = np.transpose(w)*output_hidden
        output_output = 1/(1+np.exp(-input_output))
        error = (output_var - output_output)
        d = (output_var - output_output)*np.matrix.transpose(output_output)*(1-output_output)
        y = output_hidden * np.matrix.transpose(d)
        #print y
        #print del_w
        del_w = 0.8*del_w + 0.0006*y
        e = w*d
        dd = e*np.matrix.transpose(output_hidden)*(1- output_hidden)
        x = input_input * np.matrix.transpose(dd)
        del_v = 0.8*del_v + 0.0006*x
        v = v + del_v
        w = w + del_w
        er = np.array(error)
        er = er**2
        erm = np.mean(er)
        count = count + 1
        #print float(count/10000000.0)
        print erm
        if erm<0.01:
            break
        elif count>1000000:
            break
    #print w
    #print v
    return w,v,erm,count



input_input = np.matrix([[0.4,0.3,0.6,0.2,0.4],[-0.7,-0.5,0.1,0.4,-0.2]])
output_var = np.matrix([[0.1,0.05,0.3,0.25,0.12]])
print 'start'
print input_input.shape
print output_var.shape
print 'end'

w = np.matrix(np.random.rand(3,1))
v = np.matrix(np.random.rand(2,3))


w,v,erm,count = train_data(w,v,input_input,output_var)

print w
print v
print erm
print count
