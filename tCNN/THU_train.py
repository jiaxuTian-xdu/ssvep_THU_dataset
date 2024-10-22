from keras.callbacks import ModelCheckpoint
from net import tcnn_net
import data_generator
import THU_data_generator
import scipy.io as scio
from scipy import signal
from keras.models import Model
from keras.layers import Input
import numpy as np
from random import sample
import os
import matplotlib.pyplot as plt
def get_filted_data(data,wn1,wn2):
    for i in range(40):
        for j in range (6):
            b, a = signal.butter(6, [wn1, wn2], 'bandpass')
            data[:,:,i, j] = signal.filtfilt(b, a,data[:,:,i, j])


    return data
def large_train():
    data = np.empty((10,64, 1500, 40, 6))
    for sub_selelct in range(1, 10):
        path = '/Users/ahahaha/PycharmProjects/FB-tCNN/data/THU_dataset/S%d.mat' % sub_selelct
        data[sub_selelct] = scio.loadmat(path)
        data[sub_selelct]  = data['data']
        data[sub_selelct]  = get_filted_data(data[sub_selelct],wn1,wn2)
    return data

if __name__ == '__main__':
    # open the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    fs = 1000
    channel = 9
    train_epoch = 400
    f_down = 3
    f_up = 30
    wn1 = 2 * f_down / fs
    wn2 = 2 * f_up / fs
    batchsize=512
    for sub_selelct in range(7, 8):
        path = '/Users/ahahaha/PycharmProjects/FB-tCNN/data/THU_dataset/S%d.mat' % sub_selelct
        data = scio.loadmat(path)
        data = data['data']
        data = get_filted_data(data,wn1,wn2)
        labels = np.arange(40)  # 40 个标签
        c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
        batchsize = 256
        group_indices = sample(range(6), 6)
        train_indices = group_indices[:5]
        test_index = group_indices[5]
        train_data = data[c, :, :, :][:, :, :, train_indices]
        test_data = data[c, :, :, :][:, :, :, test_index].reshape(9, 1500, 40, 1)
        train_gen = THU_data_generator.T_train_datagenerater(batchsize,labels,train_data)
        val_gen = THU_data_generator.T_val_datagenerater(batchsize, labels, test_data)
        # %% setting the input of the network
        input_shape = (channel, 800, 1)
        input_tensor = Input(shape=input_shape)
        preds = tcnn_net(input_tensor)
        model = Model(input_tensor, preds)
        # the path of the saved model and you need to change it
        model_path = '/Users/ahahaha/PycharmProjects/FB-tCNN/model/model_%d.h5' % ( sub_selelct)
        # some hyper-parameters in the training process
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='auto')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # training
        history = model.fit_generator(

            train_gen,
            steps_per_epoch=100,
            epochs=train_epoch,
            validation_data=val_gen,
            validation_steps=1,
            callbacks=[model_checkpoint]
        )
        print(sub_selelct)







