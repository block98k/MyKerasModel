from keras.datasets import mnist
from keras.utils import np_utils 
import numpy as np
def loadmnist(norm=1,one_hot=1,expand=4):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if norm:
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)  
        y_test = np_utils.to_categorical(y_test, num_classes=10)
    if expand>0:
        X_train=np.expand_dims(X_train,axis=expand)
        X_test=np.expand_dims(X_test,axis=expand)
    return (X_train, y_train), (X_test, y_test)