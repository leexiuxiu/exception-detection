# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 10:25
# @Author  : wanghelong
# @File : tu_3D.py
# @Software : PyCharm
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Embedding, Conv2D, MaxPooling2D, Reshape, BatchNormalization, Dropout, \
    Input, concatenate, GlobalAveragePooling2D, Flatten, ConvLSTM2D, ConvLSTM2DCell, LSTM, Conv3D
from keras.optimizers import adam
import numpy as np
import linecache
from keras import losses, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D

# losses.binary_crossentropy
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文


def count_line(files_in):
    file_open = open(files_in, 'r')
    count = 0
    for line in file_open:
        count += 1
    file_open.close()
    return count


def train(files_train, labels_train, model_path, files_test, label_test, predict_save):
    x_train = np.loadtxt(files_train, delimiter=',')
    y_train = np.loadtxt(labels_train, delimiter=',')
    # y_train=np.reshape(y_train,(-1,40,1))

    # --------- model structure -----------
    main_input = Input(shape=(3,), dtype='float32', name='MainInput')
    layer = Dense(10)(main_input)
    # layer=BatchNormalization()(layer)

    # layer=Dropout(0.5)(layer)
    layer = Dense(2, activation='softmax')(layer)
    # layer=LSTM(96,return_sequences=False,activation='tanh')(layer)

    output = layer

    # ------------------------
    model = Model(inputs=main_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    tensorboard_path = './'
    tbCallback = TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_grads=True,
                             write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                             embeddings_metadata=None)
    checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=0, save_best_only='True')

    model.fit(x_train, y_train, batch_size=6, epochs=80, shuffle=True, callbacks=[tbCallback, checkpoint])

    # model.fit(x_train,y_train,batch_size=12,epochs=700,shuffle=True)
    # ---------------- add test -------
    x_test = np.loadtxt(files_test, delimiter=',')
    y_test = np.loadtxt(label_test, delimiter=',')
    y_pred = model.predict(x_test)
    np.savetxt(predict_save, y_pred, fmt='%f', delimiter=',')
    loss, acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print(acc)
    # -------------------------
    # model.save(save_path)
    # return y_train


def Calculatte(pred_file, label_file):
    # file_label=open(label_file,'r')
    Drn = 0
    Dra = 0
    with open(pred_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # i:0-n
        for i, rows in enumerate(reader):
            # print (i) #row is a list
            Pred = int(rows[0])
            line = linecache.getline(label_file, i + 1)
            Label = int(line[0])
            if (Label == 0 and Pred == 0):
                Drn += 1
            if (Label == 1 and Pred == 1):
                Dra += 1
    print('Drn: ', Drn, ' Dra: ', Dra)
    return Dra + Drn


def Count_nor_ano(label):
    Num_nor = 0
    Num_ano = 0
    with open(label, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # i:0-n
        for i, rows in enumerate(reader):
            if int(rows[0]) == 0:
                Num_nor += 1
            if int(rows[0]) == 1:
                Num_ano += 1
    print('正常用户: ', Num_nor, ' 异常用户: ', Num_ano)
    return Num_ano + Num_nor


def figure_ponit():
    # normal
    x1 = []
    y1 = []
    z1 = []
    # anomalous
    x2 = []
    y2 = []
    z2 = []
    all_data_in = 'Data/Mix/' + 'Mix_all_loss2.csv'
    all_label_in = 'Data/Mix/' + 'Mix_all_label3.csv'
    line_counts = count_line(all_label_in)
    for i in range(line_counts):
        line = linecache.getline(all_data_in, i + 1)
        label = linecache.getline(all_label_in, i + 1)
        line = line.strip()
        line = line.split(',')
        label = label.strip()
        label = label.split(',')
        # print(line)
        # print(label[0])
        # exit(0)
        if label[0] == '0':
            if float(line[0])>=4.0:
                x1.append(float(line[0])-3.5)
            else:
                x1.append(float(line[0]))
            y1.append(float(line[1]))
            # m = float(line[2])
            # if m >= 4.0:
            #     print('*** '+m)
            #     m = m - 2.0
            z1.append(float(line[2]))
            # z1.append(m)
            # print (x1,y1,z1)
            # exit(0)
        else:
            # x2.append(float(line[0]))
            if float(line[0]) >= 4.0:
                x2.append(float(line[0]) - 3.5)
            else:
                x2.append(float(line[0]))
            y2.append(float(line[1]))
            # m = float(line[2])
            # if m >= 4.0:
            #     m = m - 2.0
            # # z2.append(float(line[2]))
            # z2.append(m)
            z2.append(float(line[2]))
    print(x1, y1, z1)

    fig = plt.figure()
    ax3d = Axes3D(fig)
    ax3d.scatter(x1, y1, z1, c='g', label='正常用户')
    ax3d.scatter(x2, y2, z2, c='r', label='异常用户')

    ax3d.set_zlabel('角色特征的WDD', fontdict={'size': 13, 'color': 'black'})
    ax3d.set_ylabel('行为序列的WDD', fontdict={'size': 13, 'color': 'black'})
    ax3d.set_xlabel('行为特征的WDD', fontdict={'size': 13, 'color': 'black'})

    ax3d.legend()
    # plt.savefig('Data/Mix/'+'Point4.jpg')
    plt.savefig('Data/Mix/' + 'Point6.jpg')

    plt.show()


if __name__ == "__main__":

    # -------------------------- end -----------------------------------
    # ------------- 三维散点图 (scatter diagram) ----------------
    figure_ponit()