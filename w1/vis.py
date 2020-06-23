#!/usr/bin/python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from matplotlib import pyplot
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

def load_dataset(): 
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    print("Train: X = %s, y = %s" % (trainX.shape, trainY.shape))
    print("Test: X = %s, y = %s" % (testX.shape, testY.shape))

    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    print(trainX.shape)
    return trainX, trainY, testX, testY

def prep_pixel(train, test): 
    train_norm = train.astype('float32')/255.0
    test_norm = test.astype('float32')/255.0
    return train_norm, test_norm

 load_dataset()
