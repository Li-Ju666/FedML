from pandas import read_csv
import numpy as np

csv_path = "/home/sariel/PycharmProjects/fedml/FedML/data/raw_data.csv"

data = np.array(read_csv(csv_path, error_bad_lines=False))[::,1]
data = data.astype(np.float)

sq_len = 20
result = []
for index in range(len(data) - sq_len):
    print(index)
    result.append(data[index:index+sq_len])
result = np.array(result)

np.savetxt("/home/sariel/PycharmProjects/fedml/FedML/data/train.csv",
           fmt='%.6f',X = result, delimiter=",")

