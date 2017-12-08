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
import sys
import ExtractData

# Extract the data and fill in the inputs and outputs for the AI's training
ExtractData.run()
times = array(ExtractData.times)
quants = array(ExtractData.quants)
quant_total = array(ExtractData.quant_total)
spec_pow_list = ExtractData.spec_pow
spec_pow = array(ExtractData.spec_pow)
f_initials = (ExtractData.f_initials)

def get_args():
    """
    Gets the command line arguments cleanly and error checks
    :return: the command line arguments cleaned
    """
    if len(sys.argv) != 5:
        print("Usage: AI <input_time> <input_spec_pow_of_list> <num_hidden_layers> <num_nodes_per_layer>")
        return
    inps = []
    inps.append(sys.argv[1])
    inps.append(sys.argv[2])
    inps.append(sys.argv[3])
    inps.append(sys.argv[4])
    return inps

inps = get_args()

# The inputs into the model
input_time = inps[0]
input_spec_pow = spec_pow[inps[1]]
inputs = f_initials.append(input_time)
inputs = f_initials.append(input_spec_pow)
inputs = array(inputs)
output = array(quants[inps[1]][input_time])

num_layers = inps[3]
layer_size = inps[4]

# The number of nodes (DEFAULT) in each of the hidden layers initially
num_nodes_hl1 = 30
num_nodes_hl2 = 30
num_nodes_hl3 = 30

x = tf.placeholder(tf.float32, [None, len(inputs)])
y = tf.placeholder(tf.float32, [None, len(output)])


def layering(x, input_size, output_size):
    """
    Creating a layer based on variable input
    :param x: input
    :param input_size: the input size specified to the layer
    :param output_size: the output size specified to the layer
    :return: a layer in the form of multiplying the weights plus a bias
    """
    w = tf.Variable(tf.truncated_normal(shape=[input_size, output_size]))
    b = tf.Variable(tf.zeros([output_size]))
    return tf.sigmoid(tf.matmul(x, w) + b)

def create_variable_layers():
    """
    Creating the required number of hidden layers with the specified size
    """
    hidden = x
    for i in range(num_layers):
        hidden = layering(hidden, input_size, layer_size[i])
        input_size = layer_size[i]

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
