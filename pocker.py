from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np



np.random.seed(42)
X_train = X_train.reshape(60000, 784)

Y_train = np_utils.to_categorical(y_train, 10)