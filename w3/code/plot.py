import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def get_path():
    path = '/home/sariel/PycharmProjects/fedml/FedML/w3/code'
    return path+'/central_result.csv', path+'/fed_result.csv'

def get_data(path):
    data = np.array(read_csv(path))
    data = data.reshape(data.shape[0]*data.shape[1])
    return data


cent_path, fed_path = get_path()
cent_data = get_data(cent_path)
fed_data = get_data(fed_path)

##baseline = cent_data[int(0.5*len(cent_data)):].mean()
##plt.plot([0, len(fed_data)], [baseline, baseline], label = "Centralized")
plt.plot(cent_data, label = "Federated")
plt.plot(fed_data, label = "Centralized")

plt.xlabel('Num of runs')
plt.ylabel('Accuracy/%')
plt.title('MNIST (fully connected network): accuracy ~ num of runs')
plt.legend(loc = 1)
plt.ylim(70,95)
##plt.show()
plt.savefig('cmp.png')