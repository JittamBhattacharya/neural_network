import numpy as np

def image_to_pcaimage(x,y):
    n_samples , height , width = x.shape
    xx = []
    yy = []
    for i in range(n_samples):
        xx = xx + list(x[i])
        yy = yy + list(y[i]*np.ones(height))
    xx = np.array(xx)
    yy = np.array(yy)
    return xx,yy
