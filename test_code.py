import numpy as np
import matplotlib.pyplot as plt
import my_data_set
import cv2

img_array , distance_array = my_data_set.fetch_data()
# printing the mean array
#print img_array
#print distance_array
res_array =[]

for img in img_array:
    #print img.shape
    image_x , image_y = img.shape
    fax, fay = 300.0/float(image_x) , 300.0/float(image_y)
    #print fax,fay
    res = cv2.resize(img,None,fx= fay, fy= fax, interpolation = cv2.INTER_LINEAR)
    img = res
    if res.shape == (300,300):
        res_array.append(res)

res_array = np.array(res_array)
#print res_array

#print res_array[0]

mean_image = np.mean( res_array , axis = 0)

plt.figure
plt.imshow(mean_image.reshape((300,300)) , cmap = plt.cm.gray)
plt.show()

from sklearn import datasets,neighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA

n_samples = len(res_array)
#print n_samples

distance_array = np.array(distance_array)
#print distance_array

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

x , y = image_to_pcaimage(res_array , distance_array)

print x.shape
print y.shape
#print x
#print y
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.333 ,random_state = 42)
print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape

def plot_gallery(images,h,w,titles = None,n_rows =3 ,n_cols = 4):
    plt.figure(figsize = (1.8*n_cols,2.4*n_rows))
    plt.figure(figsize=(1.8 * n_cols, 2.4 * n_rows))
    plt.subplots_adjust(bottom = 0 , left = 0.01 , right = 0.99 , top = 0.90)# , nspace = 0.25)
    for  i in range(n_rows*n_cols):
        #print 'IN'
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap = plt.cm.gray)
        if titles != None :
            plt.title(titles[i],size = 12)
    #print 'in'
    plt.show()

labels = np.arange(12)
#print labels

titles = ['bucket#' + str(i) for i in labels]

#plot_gallery(x,300,300,titles)

# Calculate a  set of eigen facesgen-face
# Reduce the dimensionality of the feature space

n_components  = 30
height , width = 300 ,300


pca = RandomizedPCA(n_components  = n_components , whiten = True  ) .fit(x_train)
# Find the eigen values of the feature space

eigen_faces = pca.components_.reshape(pca.components_.shape)

# plot_gallery(eigen_faces,height,width,4,2,titles)

# projecting the data onto the eigen spaces

x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

#print x_train_pca.shape()

# Use a KNN Classifier in the transformed space to identify the object50
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(x_train_pca,y_train)

# Detect the faces in the test_set

y_pred_test = knn_classifier.predict(x_test_pca)
correct_count = 0.0

for i in range(len(y_test)):
    if y_pred_test[i] == y_test[i]:
        correct_count = correct_count + 1.0

accuracy =  correct_count/ float(len(y_test))

print accuracy

y_train = y_train /float(max(distance_array))
print x_train_pca
print y_train

l = np.exp(-8)
def train_data(w,v,input_var,output_var):
    count = 0
    del_w = np.matrix(np.zeros((5000,1)))
    #print del_w
    del_v = np.matrix(np.zeros((30,5000)))
    while 1:
        input_hidden = np.matrix.transpose(v)*input_input
        #print input_hidden
        output_hidden = 1/(1+np.exp(-l*l*input_hidden))
        #print output_hidden
        input_output = np.transpose(w)*output_hidden
        output_output = 1/(1+np.exp(-l*input_output))
        error = (output_var - output_output)
        d = (output_var - output_output)*np.matrix.transpose(output_output)*(1-output_output)
        y = output_hidden * np.matrix.transpose(d)
        #print y
        #print del_w
        del_w = 0.8*del_w + 0.6*y
        e = w*d
        dd = e*np.matrix.transpose(output_hidden)*(1- output_hidden)
        x = input_input * np.matrix.transpose(dd)
        del_v = 0.8*del_v + 0.6*x
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
        elif count>5:
            break
    #print w
    #print v
    return w,v,erm,count

input_input = np.matrix(x_train_pca.reshape((30,10605)))
output_var = np.matrix(y_train)
print 'start'
print input_input.shape
print output_var.shape
print 'end'

w = np.matrix(np.random.rand(5000,1))
v = np.matrix(np.random.rand(30,5000))




w,v,erm,count = train_data(w,v,input_input,output_var)

print w
print v
print erm
print count


print x_test_pca.shape
print y_test.shape

input_input = np.matrix(x_test_pca.reshape((30,5295)))
output_var = np.matrix(y_test.reshape((1,5295)))
w,v,erm,count = train_data(w,v,input_input,output_var/float(max(distance_array)))

print w
print v
print erm
print count
