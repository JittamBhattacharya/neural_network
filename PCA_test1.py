import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets,neighbors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA


faces_data = fetch_olivetti_faces()
n_samples, height , width = faces_data.images.shape

x = faces_data.data
n_features = x.data[1]
y = faces_data.target
n_class =  int(max(y)+1)

#Shuffle the data randomly and make test train split

print x
print y

print x.shape
print y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 42)
print x_train.shape
print y_train.shape
print x_test.shape
print y_test.shape
mean_image = np.mean(x_train,axis = 0)
plt.figure
plt.imshow(mean_image.reshape((64,64)),cmap = plt.cm.gray)
plt.show()

# make  a function to visualize set of images as a function

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
# Visualize some faces from the training set

chosen_images = x_train[:12]
chosen_labels = y_train[:12]

titles = ['Person#' + str(i) for i in chosen_labels]
plot_gallery(chosen_images,height,width,titles)

# Calculate a  set of eigen facesgen-face
# Reduce the dimensionality of the feature space

n_components  = 50
print x_train.shape



pca = RandomizedPCA(n_components  = n_components , whiten = True  ) .fit(x_train)
# Find the eigen values of the feature space

eigen_faces = pca.components_.reshape((n_components,height,width))

# Visualize the eigen faces
titles = ['eigen_face# ' + str(i) for i in range(12)]
plot_gallery(eigen_faces,height,width,titles)

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
