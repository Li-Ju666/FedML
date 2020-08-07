from keras.datasets import cifar10
import numpy
(trainx, trainy), (_, _) = cifar10.load_data()
data = trainx.reshape(50000, 32*32*3)
print(data.shape)

print((trainx == data.reshape(50000, 32, 32, 3)).all())

data = numpy.column_stack((trainy, data))
print(data.shape)

numpy.savetxt("/home/sariel/PycharmProjects/fedml/FedML/data/train.csv", data, delimiter=",")