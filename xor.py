from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

X = np.array([[0,0],[0,1],[0,1],[1,1]])
y = np.array([[0, 0, 0],[1, 1, 0],[1, 1, 0],[0, 1, 1]])

model = Sequential()
model.add(Dense(4, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(3))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, nb_epoch=1000)
print(model.predict_proba(X))