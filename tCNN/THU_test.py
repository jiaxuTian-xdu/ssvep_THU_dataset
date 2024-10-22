import keras
from keras.models import load_model
import scipy.io as scio
import random
import numpy as np
from scipy import signal
import os



# generating the test samples, more details refer to data_generator
def datagenerator(batchsize,labels,train_data):
    labels = list(labels)
    x_train, y_train = list(range(batchsize)), list(range(batchsize))
    for i in range(batchsize):
        k1 = random.sample(labels, 1)[0]
        k2 = random.sample(range(5), 1)[0]
        start = random.randint(0, 700)
        x_1 = train_data[:, start:start + 800, k1, k2]
        x_2 = np.reshape(x_1, (9, 800, 1))
        x_train[i] = x_2
        y_train[i] = keras.utils.to_categorical(k1, num_classes=40, dtype='float32')
    x_train = np.reshape(x_train, (256, 9, 800, 1))
    y_train = np.reshape(y_train, (256, 40))
    return x_train, y_train

def get_filted_data(data,wn1,wn2):

    for i in range(40):
        for j in range (6):
            b, a = signal.butter(6, [wn1, wn2], 'bandpass')
            data[:,:,i, j] = signal.filtfilt(b, a,data[:,:,i, j])

    return data

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    # Setting hyper-parameters, more details refer to "tCNN_train"
    down_sample = 4
    fs = 1000 / down_sample
    channel = 9
    train_epoch = 100
    batchsize = 1000
    f_down = 3
    f_up = 50
    wn1 = 2 * f_down / fs
    wn2 = 2 * f_up / fs

    total_av_acc_list = []
    for sub_selelct in range(1, 10):
        # the path of the dataset and you need to change it for your test
        path = '/Users/ahahaha/PycharmProjects/FB-tCNN/data/THU_dataset/S%d.mat' % sub_selelct
        data = scio.loadmat(path)
        data = data['data']
        data=get_filted_data(data,wn1,wn2)
        labels = np.arange(40)  # 40 个标签
        c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
        batchsize = 256
        group_indices = random.sample(range(6), 6)
        train_indices = group_indices[:6]
        train_data = data[c, :, :, :][:, :, :, train_indices]
        av_acc_list = []
        model_path = '/Users/ahahaha/PycharmProjects/FB-tCNN/model/model_7.h5'
        # load the model
        model = load_model(model_path)
        print("load successed")
        print( sub_selelct)
        acc_list = []
        # test 5 times and get the average accrucy of the 5 times as the test result, here you can only test once
        for j in range(5):
            # get the filtered EEG-data, label and the start time of the test samples, and the number of the samples is "batchsize"
            x_train, y_train = datagenerator(batchsize, labels, train_data)
            a, b = 0, 0
            # get the predicted results of the batchsize test samples
            y_pred = model.predict(np.array(x_train))
            true, pred = [], []
            y_true = y_train
            # Calculating the accuracy of current time
            for i in range(batchsize - 1):
                y_pred_ = np.argmax(y_pred[i])
                pred.append(y_pred_)
                y_true1 = np.argmax(y_train[i])
                true.append(y_true1)
                if y_true1 == y_pred_:
                    a += 1
                else:
                    b += 1
            acc = a / (a + b)
            acc_list.append(acc)
        av_acc = np.mean(acc_list)
        print(av_acc)
        av_acc_list.append(av_acc)
    total_av_acc_list.append(av_acc_list)

