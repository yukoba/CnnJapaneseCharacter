# TFlearn version of learn.py
# Run with tflearn 0.2.1 and tensorflow 0.10.0

import numpy as np
import scipy
from sklearn.cross_validation import train_test_split
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.initializations import normal
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical

nb_classes = 72
# input image dimensions
img_rows, img_cols = 64, 64
# img_rows, img_cols = 127, 128

ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * 160):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
    # X_train[i] = ary[i]
Y_train = np.repeat(np.arange(nb_classes), 160)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)

network = input_data(shape=[None, img_rows, img_cols, 1])


def m6_1():
    global network
    my_init = normal(stddev=0.1)
    network = conv_2d(network, 32, 3, weights_init=my_init, activation='relu')
    network = conv_2d(network, 32, 3, weights_init=my_init, activation='relu')
    network = max_pool_2d(network, 2)
    network = dropout(network, 0.5)

    network = conv_2d(network, 64, 3, weights_init=my_init, activation='relu')
    network = conv_2d(network, 64, 3, weights_init=my_init, activation='relu')
    network = max_pool_2d(network, 2)
    network = dropout(network, 0.5)

    network = flatten(network)
    network = fully_connected(network, 256, weights_init=my_init, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, nb_classes, activation='softmax')


def classic_neural():
    global network
    network = flatten(network)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, nb_classes, activation='softmax')


m6_1()
# classic_neural()

network = regression(network, optimizer='adagrad', loss='categorical_crossentropy', learning_rate=0.01)
model = tflearn.DNN(network, checkpoint_path='model_vgg', max_checkpoints=1, tensorboard_verbose=0)
model.fit(X_train, Y_train,
          validation_set=(X_test, Y_test),
          n_epoch=100, shuffle=True,
          show_metric=True, batch_size=16, snapshot_step=X_train.shape[0] // 16,
          snapshot_epoch=False, run_id='vgg_hiragana')
