import keras.models
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy

def read_data(filename, sample_fraction=1.0):
    """ Helper function to read and preprocess data for training with Keras. """

    data = numpy.array(pd.read_csv(filename))
    X = data[:, 1::]
    y = data[:, 0]

    # The entire dataset is 60k images, we can subsample here for quicker testing.
    if sample_fraction < 1.0:
        foo, X, bar, y = train_test_split(X, y, test_size=sample_fraction)


    # The data, split between train and test sets
    X = X.reshape((X.shape[0], 32, 32, 3))
    X = X.astype('float32')
    X /= 255
    y = to_categorical(y)
    return  (X, y)


num_models = 50
output = open("/home/sariel/PycharmProjects/fedml/FedML/w5/FMNIST/fed_result.txt", "a")
testX, testY = read_data("/home/sariel/PycharmProjects/fedml/FedML/data/data1/test.csv", sample_fraction=1)

for i in range(num_models):
    model_name = "global_model_"+str(i+1)+".h5"
    model = keras.models.load_model("/home/sariel/PycharmProjects/fedml/FedML/w5/CIFAR/models/"+model_name)
    _, x, _, y = train_test_split(testX, testY, test_size=0.5)
    _, accuracy = model.evaluate(x, y, verbose = 0)
    print(accuracy)
    output.write("%f\n"%accuracy)

output.close()
