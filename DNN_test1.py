import numpy as np
import tflearn

# Download the titanic datasets

from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load the csv file, indicate that the first column indicate the labels

from tflearn.data_utils import load_csv
data , labels = load_csv('titanic_dataset.csv',target_column = 0 , categorical_labels = True , n_classes  = 2)

#Preprocessing the data

def preprocess(data , columns_to_ignore):
    for id in sorted(columns_to_ignore,reverse = True ):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        data[i][1]  = 1.0 if data[i][1] == 'female' else 0
    return np.array(data , dtype = np.float32)

to_ignore = [1,6]
data = preprocess(data,to_ignore)

net = tflearn.input_data(shape = [None,6])
net = tflearn.fully_connected(net , 32)
net = tflearn.fully_connected(net , 32)
net = tflearn.fully_connected(net , activation = 'softmax')

net = tflearn.regression(net)
model = tflearn.DNN(net)

# start the training , apply the gradient descent algorithm

model.fit(data , labels , n_epoch = 10 , batch_size = 10 , show_metric = True)
