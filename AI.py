"""
Used to train and test the AI.
@input:
    inv fields (quantity) at time 0, as long as they are not 0
    Specific power of the data
    The time you want quantities in

@output:
    inv fields (quanitity) at time t given
@author: Virat Singh
"""

import tensorflow as tf
from numpy import array
import ExtractData

# Extract the data and fill in the inputs and outputs for the AI's training
ExtractData.run()
times = array(ExtractData.times)
quants = array(ExtractData.quants)
quant_total = array(ExtractData.quant_total)
spec_pow_list = ExtractData.spec_pow
spec_pow = array(ExtractData.spec_pow)
f_initials = (ExtractData.f_initials)

# The inputs into the model
input_time = 30
input_spec_pow = spec_pow[0]
inputs = f_initials.append(input_time)
inputs = f_initials.append(input_spec_pow)
inputs = array(input)
output = array(quants[0][input_time])

# The number of nodes in each of the hidden layers initially
num_nodes_hl1 = 30
num_nodes_hl2 = 30
num_nodes_hl3 = 30

x = tf.placeholder(tf.float32, [None, len(inputs)])
y = tf.placeholder(tf.float32, [None, len(output)])

def create_model(data):
    """
    Creates the initial neural network model.
    :param data: the input data
    :return: the predicted output
    """

    # Creating each hidden layer with the appropriate size
    hl1 = {'weights': tf.Variable(tf.random_normal([len(data), num_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl1]))}

    hl2 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl2]))}

    hl3 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl2, num_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([num_nodes_hl3, 1])),
                    'biases': tf.Variable(tf.random_normal([len(output)]))}

    # Feed forward, input data alongside the weights
    l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
    l3 = tf.nn.relu(l3)

    # Output = sum of all values times weights, plus the bias
    out = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return out

def train(data):
    """
    Trains the model based on some data
    :param data: input data
    :return: the accuracy of the training
    """

    prediction = create_model(data)
    # May want to change cost to be unique for regression, instead of classification
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # The number of cycles of training, i.e, feed forward + back propagation
    num_cycles = 5

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for cycle in num_cycles:
            loss = 0
            for i in range(len(data)):
                # Train next batch here
                i, c = sess.run([optimizer, cost], feed_dict={x: x, y: y})
                loss += c
            print("Cycle #" + cycle + " / " + num_cycles + ", loss = " + loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy = " + accuracy)

    return accuracy

train(inputs)