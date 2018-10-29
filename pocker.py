from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np


X_train = []
y_train = []
np.random.seed(42)
X_train = X_train.reshape(60000, 784)

Y_train = np_utils.to_categorical(y_train, 10)


model = Sequential()

model.add(Dense(85, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])