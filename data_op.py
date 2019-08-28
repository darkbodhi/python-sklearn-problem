import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.neural_network import MLPClassifier

# pandas and numpy for dataframes, sklearn for ML
# the prediction is based on neural networks (sklearn's ones)

i=0
for df in pd.read_excel('./test_task_DS.xlsx', chunksize=60845):
    df.to_excel('train_data.xlsx'.format(i), index=False)
    i += 1

i=60846
for df in pd.read_excel('./test_task_DS.xlsx', chunksize=26077):
    df.to_excel('test_data.xlsx'.format(i), index=False)
    i += 1

# splitting initial file into two new ones, one for test and another one for train data

data_set_train = pd.read_csv('train_data.xlsx', sep=',', header=0)
data_set_test = pd.read_csv('test_data.xlsx', sep=',', header=0)

y_tr = data_set_train.iloc[:, 0]
X_tr = data_set_train.iloc[:, 1:]

y_test = data_set_test.iloc[:, 0]
X_test = data_set_test.iloc[:, 1:]

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_tr, y_tr)
NN.predict(X_test)
resulting_data = round(NN.score(X_test, y_test), 7)

# basically processing data

exit_data_frame = pd.DataFrame(resulting_data)
df(exit_data_frame['case_size', 'id' 'pack_deck', 'wholesale_price', 'description'])

# creating a resulting DataFrame
