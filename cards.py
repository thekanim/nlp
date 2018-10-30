from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD
import numpy as np

FILE = 'poker-hand-training-true.data'

def get_input_data():
    with open(FILE, 'r') as file:
        array_input = []
        for line in file.readlines():
            line_input = []
            line = line.split(',')
            for i in range(0, len(line) - 2, 2):
                line_input.append([int(line[i]), int(line[i + 1])])
            array_input.append(line_input)
    return np.array(array_input)

def get_output_data():
    with open(FILE, 'r') as file:
        array_output = []
        for line in file.readlines():
            line = line.split(',')
            etalon = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
            last_el = int(line[-1])
            etalon[last_el] = [1]
            array_output.append(etalon)
    return np.array(array_output)


X = get_input_data()
y = get_output_data()

model = Sequential()
model.add(Dense(10, input_shape=(2, 5)))
model.add(Activation('linear'))
#model.add(Reshape((10,1)))
model.add(Dense(10))
model.add(Activation('linear'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)


print(model.summary())


model.fit(X, y, batch_size=1, epochs=200)
print(model.predict_proba(X))