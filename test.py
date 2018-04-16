import tflearn

import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

print(len(X))
print(len(Y))
print(len(testX))
print(len(testY))
