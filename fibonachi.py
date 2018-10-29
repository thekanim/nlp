from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np


X = np.array([ [1], [2], [3], [5], [8], [13], [21], [34], [55], [89], [144], [233], [377], [610], [987], [1597], [2584], [4181], [6765]])
y = np.array([ [2], [3], [5], [8], [13], [21], [34], [55], [89], [144], [233], [377], [610], [987], [1597], [2584], [4181], [6765], [10946]])


model = Sequential()
model.add(Dense(8, input_dim=1))
model.add(Activation('linear'))
model.add(Dense(8))
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('linear'))

sgd = SGD(lr=0.5)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, nb_epoch=10)
print(model.predict_proba(X))