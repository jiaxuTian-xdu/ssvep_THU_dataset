
from scipy import signal
import keras.utils
import numpy as np
from random import sample, randint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import scipy.io as scio

def T_train_datagenerater(batchsize,labels,train_data):
    labels = list(labels)

    while True:
        x_train, y_train = list(range(batchsize)),list(range(batchsize))
        for i  in range(batchsize):
            k1 = sample(labels,1)[0]
            k2 = sample(range(5),1)[0]
            start = randint(0, 700)
            x_1 = train_data[:,start:start + 800,k1,k2]
            x_2 = np.reshape(x_1,(9,800,1))
            x_train[i]= x_2
            y_train[i]= keras.utils.to_categorical(k1,num_classes=40,dtype='float32')
        x_train=np.reshape(x_train,(batchsize,9,800,1))
        y_train=np.reshape(y_train,(batchsize,40))

        yield x_train,y_train
def T_val_datagenerater(batchsize,labels,train_data):
    labels = list(labels)

    while True:
        x_train, y_train = list(range(batchsize)),list(range(batchsize))
        for i  in range(batchsize):
            k1 = sample(labels,1)[0]
            k2 = sample(range(1),1)[0]
            k3 = sample(range(10), 1)[0]
            start = randint(0, 700)
            x_1 = train_data[:, start:start + 800, k1, k2]
            x_2 = np.reshape(x_1, (9, 800, 1))
            x_train[i]= x_2
            y_train[i]= keras.utils.to_categorical(k1,num_classes=40,dtype='float32')
        x_train=np.reshape(x_train,(batchsize,9,800,1))
        y_train=np.reshape(y_train,(batchsize,40))

        yield x_train,y_train

