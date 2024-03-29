from __future__ import print_function

import numpy as np
import tflearn

# Download the Titanic dataset
import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
import load_csv
data, labels = load_csv.load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)
#print (data , labels)


# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
#print (data , labels)
data = preprocess(data, to_ignore)
#print (data , labels)
# Build neural network
#print data
tflearn.init_graph( num_cores = 4)
net = tflearn.input_data(shape=[None, 6])
print(net)
net = tflearn.fully_connected(net, 32 , activation = 'relu')
print(net)
net = tflearn.fully_connected(net, 32 , activation = 'relu')
print(net)
net = tflearn.fully_connected(net, 2, activation='softmax')
print(net)
net = tflearn.regression(net)# , loss = 'catergorical_crossentropy' , optimizer = 'adam')
print(net)
#print net
# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
