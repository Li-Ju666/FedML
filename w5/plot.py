import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np

def get_path(name):
    path = '/home/sariel/PycharmProjects/fedml/FedML/w5/'
    return path+name+'/central_result.txt', path+name+'/fed_result.txt'

def get_data(path):
    data = np.array(read_csv(path))
    data = data.reshape(data.shape[0]*data.shape[1])
    return data

name = 'Time'
cent_path, fed_path = get_path(name)
cent_data = get_data(cent_path)
fed_data = get_data(fed_path)

baseline = cent_data[int(0.5*len(cent_data)):].mean()
plt.plot([0, len(fed_data)], [baseline, baseline], label = "Centralized")
plt.plot(fed_data, label = "Federated")
plt.text(0,1.5,'1.759/Celsius')
plt.xlabel('Num of runs')
plt.ylabel('Mean absolute error/Celsius')
plt.title('Temperature in Melbourne (LSTM): MAE ~ num of runs')
plt.legend(loc = 1)
plt.ylim(1,4)
##plt.show()
plt.savefig(name+'.png')