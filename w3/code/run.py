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


def normalize(data):
    data_norm = data.astype('float32')
    data_norm = data_norm/255.0
    return data_norm

def load_data():
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    trainX = trainX.reshape((trainX.shape[0], 28*28))
    testX = testX.reshape((testX.shape[0], 28*28))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    trainX, testX = normalize(trainX), normalize(testX)

    return trainX, trainY, testX, testY

def partition_data(trainX, trainY, num):
    ran_order = sample(range(0, trainX.shape[0]), trainX.shape[0])
    local_size=int(trainX.shape[0]/num)
    partitionedX=[]
    partitionedY=[]
    for i in range(0,num):
        partitionedX.append(trainX[ran_order[i*local_size:(i+1)*local_size]])
        partitionedY.append(trainY[ran_order[i*local_size:(i+1)*local_size]])

    return numpy.array(partitionedX), numpy.array(partitionedY)

def save_data(partitionedX, partitionedY, testX, testY):
    for i in range(0, partitionedX.shape[0]):
        numpy.savetxt("./data/localX_"+str(i)+".csv", partitionedX[i], fmt="%i", delimiter=",")
        numpy.savetxt("./data/localY_"+str(i)+".csv", partitionedY[i], fmt="%i", delimiter=',')
    numpy.savetxt("./data/testX.csv", testX, fmt="%i", delimiter=",")
    numpy.savetxt("./data/testY.csv", testY, fmt="%i", delimiter=",")

def define_model():
        model = Sequential()
        model.add(Dense(100, input_dim = 28*28, activation = 'relu', kernel_initializer='he_uniform'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(10, activation = 'softmax'))

        opt = SGD(lr = 0.01, momentum = 0.9)
        model.compile(optimizer = opt, loss = 'categorical_crossentropy',
                        metrics = ['accuracy'])
        return model


def fed_learn():
    num_clients = 5
    num_run = 10
    num_epoch = 5
    bs = 10

    trainX, trainY, testX, testY = load_data()
    X, Y = partition_data(trainX, trainY, num_clients)

    ## build global model
    global_model = define_model()
    global_weights = global_model.get_weights()

    for run_round in range(0, num_run):
        print("This is run ", run_round)
        local_weights_list = list()

        for client in range(0, num_clients):
            ##print("For client ", client)
            local_model = define_model()
            local_model.set_weights(global_weights)
            local_model.fit(X[client], Y[client], epochs=num_epoch, batch_size=bs, verbose=0)
            ##_, accuracy = local_model.evaluate(X[client], Y[client], verbose=0)
            ##print("Accuracy: %.2f"%(accuracy*100))

            local_weights_list.append(local_model.get_weights())

        global_weights = numpy.mean(local_weights_list, axis=0)
        global_model.set_weights(global_weights)
        _, accuracy = global_model.evaluate(testX, testY, verbose=0)
        print("Global model accuracy: %.2f" % (accuracy * 100))

fed_learn()