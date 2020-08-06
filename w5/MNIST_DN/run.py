#!/usr/bin/python3
import sys
import warnings
warnings.simplefilter(action='ignore')

from keras.datasets import fashion_mnist
import numpy
from keras.utils import to_categorical
from random import sample
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import numpy

def read_data(filename, sample_fraction=1.0):

    data = numpy.array(pd.read_csv(filename))
    X = data[:int(0.8*data.shape[0]), 1::]
    y = data[:int(0.8*data.shape[0]), 0]

    # The entire dataset is 60k images, we can subsample here for quicker testing.
    if sample_fraction < 1.0:
        foo, X, bar, y = train_test_split(X, y, test_size=sample_fraction)


    # The data, split between train and test sets
    X = X.reshape((X.shape[0], 28 * 28))
    X = X.astype('float32')
    X /= 255
    y = to_categorical(y)
    return  (X, y)


def define_model():
        model = Sequential()
        model.add(Dense(100, input_dim = 28*28, activation = 'relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(10, activation = 'softmax'))

        opt = SGD(lr = 0.01, momentum = 0.9)
        model.compile(optimizer = opt, loss = 'categorical_crossentropy',
                        metrics = ['accuracy'])
        return model

def centralized_learn(model, trainX, trainY, testX, testY, file):
    bs=10
    model.fit(trainX, trainY, epochs = 1, batch_size=bs, verbose=0)
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    print(accuracy)
    file.write("%f\n"%accuracy)

def run(num):
    model = define_model()
    output = open("./central_result.txt", "a")
    for i in range(num):
        trainX, trainY = read_data("/home/sariel/PycharmProjects/fedml/FedML/data/train.csv", sample_fraction=0.1)
        testX, testY = read_data("/home/sariel/PycharmProjects/fedml/FedML/data/data1/test.csv", sample_fraction=0.1)
        centralized_learn(model, trainX, trainY, testX, testY, output)
    ##fed_learn(trainX, trainY, testX, testY)
    output.close()
run(50)