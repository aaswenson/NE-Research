from __future__ import division, print_function, absolute_import

import tflearn

import numpy as np

dimensions = ['AR', 'core_r', 'cool_r', 'PD', 'power', 'enrich']
fit_dims = ['core_r', 'PD', 'enrich']

# Data loading and preprocessing
def load_from_csv(datafile="depl_results.csv"):
    """load the results data from a csv.
    """
    data = np.genfromtxt(datafile, delimiter=',',
            names=dimensions + ['keff', 'ave_E', 'mass'])
    
    return data

dep = []
ind = []

data = load_from_csv()

for idx, item in enumerate(data):
    res = []
    for dim in fit_dims:
        res.append(data[idx][dim])
    dep.append(res)
    ind.append(data[idx]['keff'])


X = dep[:2000]
Y = ind[:2000]
testX = dep[2000:]
testY = ind[2000:]




# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 3])
dense1 = tflearn.fully_connected(input_layer,2, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 2, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 1, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")
