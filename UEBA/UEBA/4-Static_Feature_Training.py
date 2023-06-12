from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Multiply,Activation,Embedding,Conv2D,MaxPooling2D,Reshape,BatchNormalization,Dropout,Input,concatenate,GlobalAveragePooling2D,Flatten,ConvLSTM2D,ConvLSTM2DCell,LSTM,Conv3D,Conv1D
from keras.optimizers import adam
import numpy as np 
import linecache
import tensorflow as tf 
import os
from keras import losses,metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

def path_check(path):
    folders=os.path.exists(path)
    if not folders:
        print("Create new folders: ' %s ' successfully!"%(path))
        os.makedirs(path)
    else:
        print("The folder  ' %s ' exits, Please check!"%(path))
        os._exit(0)

def count_line(files_in):
    file_open=open(files_in,'r')
    count=0
    for line in file_open:
        count+=1
    return count   

def data_generator(files_name,num_days=5,batch_size=30):
    # 删除最后一行数据，因为可能会有缺漏
    counts=count_line(files_name)-1
    real_counts=counts-num_days+1
    rounds=int(real_counts/batch_size)
    # print (real_counts)
    while (1):   
        for i in range(rounds):
            data_batch=[]
            label_batch=[]
            for j in range (batch_size):
                data_index=i+j+1
                label_index=data_index+4
                data_one=linecache.getline(files_name,data_index).strip().split(',')+linecache.getline(files_name,data_index+1).strip().split(',')+linecache.getline(files_name,data_index+2).strip().split(',')+linecache.getline(files_name,data_index+3).strip().split(',')
                labe_data=linecache.getline(files_name,label_index).strip().split(',')
                data_batch.append(data_one)
                label_batch.append(labe_data)
            # print(np.shape(labe_data))
            # print (data_one)
            # exit (0)
            data_batch=np.array(data_batch)
            label_batch=np.array(label_batch)
            data_out=np.reshape(data_batch,(batch_size,192))
            label_out=np.reshape(label_batch,(batch_size,48))
            i=i+batch_size+1
            yield data_out,label_out



def train(files_train,labels_train,save_path,files_test,predict_save):
    """ Deep Learning model for users' features
    # Params:
    ----
        files_train: training data (data for model training)
        labels_train: labels of training data
        save_path: the path of model that is going to be saved 
        files_test: test data 
        predict_save: the predicted labels of test data 
        
    """
    # ---------------模型输入 (input)---------------
    x_train=np.loadtxt(files_train,delimiter=',')
    y_train=np.loadtxt(labels_train,delimiter=',')
    # x_train=x_train.reshape(1,x_train.shape[0],x_train.shape[1],1)
    # y_train=y_train.reshape(1,y_train.shape[0],y_train.shape[1],1)
    # ---------------模型结构 (model structure)---------------
    main_input=Input(shape=(148,),dtype='float32',name='MainInput')
    Inputs=Reshape((4,37))(main_input)
    # Inputs=Reshape((1,1,4,37))(Inputs)
    layer=LSTM(100,return_sequences=True,activation='tanh')(Inputs)
    layer=LSTM(116,return_sequences=False,activation='tanh')(layer)
    # conv_lstm_1 = ConvLSTM2D(filters=24, kernel_size=(3, 3),
    #                          dropout=0.5)(Inputs)
    # conv_lstm_2 = ConvLSTM2D(filters=128, kernel_size=(3, 3),
    #                          dropout=0.5)(conv_lstm_1)
    # conv_lstm_3 = ConvLSTM2D(filters=48, kernel_size=(3, 3),
    #                          dropout=0.5)(conv_lstm_2)
    # attention
    attention_probs = Dense(116, activation='softmax', name='attention_vec')(layer)
    attention_mul = Multiply()([layer, attention_probs])
    # ATTENTION PART FINISHES HERE
    # attention_mul = Dense(64)(attention_mul)  # 原始的全连接

    layer = Dense(37, activation='relu')(attention_mul)
    layer=Dense(37,activation='relu')(layer)
    # layer=Dense(48,activation='relu')(conv_lstm_3)
    output=layer
    model=Model(inputs=main_input,outputs=output)
    model.compile(optimizer='adam',loss='mse',metrics=['binary_accuracy'])

    # ------------------模型保存 (save the model)------------------
    tensorboard_path='Data/'+ USERNAME +'/Model/Feature'
    tbCallback=TensorBoard(log_dir=tensorboard_path,histogram_freq=0,write_graph=True,write_grads=True,write_images=True,embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    checkpoint=ModelCheckpoint(save_path,monitor='loss',verbose=0,save_best_only='True')
    model.fit(x_train,y_train,batch_size=6,epochs=30,shuffle=True,callbacks=[tbCallback,checkpoint])
    
    # -----------------数据预测 (data prediction )---------------------
    x_test=np.loadtxt(files_test,delimiter=',')
    y_pred=model.predict(x_test)
    np.savetxt(predict_save,y_pred,fmt='%f',delimiter=',')


def retrain(files_train,labels_train,save_path,files_test,predict_save):
    """
    model retraining 
    """
    x_train=np.loadtxt(files_train,delimiter=',')
    y_train=np.loadtxt(labels_train,delimiter=',')
    model=load_model(save_path)

    # ----------------- load the model-----
    tensorboard_path='Data/'+ USERNAME +'/Model/Feature'
    tbCallback=TensorBoard(log_dir=tensorboard_path,histogram_freq=0,write_graph=True,write_grads=True,write_images=True,embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    checkpoint=ModelCheckpoint(save_path,monitor='loss',verbose=0,save_best_only='True')

    model.fit(x_train,y_train,batch_size=6,epochs=800,shuffle=True,callbacks=[tbCallback,checkpoint])
    
    # ---------------- add test -------
    x_test=np.loadtxt(files_test,delimiter=',')
    y_pred=model.predict(x_test)
    np.savetxt(predict_save,y_pred,fmt='%f',delimiter=',')

# ----------------------------- test ----------------
def test(files_test,save_path,predict_save):
    
    # labels_name='Data/'+username+'/feature/label_test.csv'
    x_test=np.loadtxt(files_test,delimiter=',')
    model=load_model(save_path)
    pred=model.predict(x_test)
    print(np.shape(pred))
    # pred 矩阵中如果大于 0.5 保持不变，否则置 0）
    pred=np.where(pred>0.47,pred,0)
    np.savetxt(predict_save,pred,fmt='%f',delimiter=',')
    return pred


#  define my loss
def my_loss_forFeatures(label_test,predict_save,myloss_save,figure_save):
    y_true=np.loadtxt(label_test,delimiter=',')
    y_pred=np.loadtxt(predict_save,delimiter=',')
    batch_size=np.shape(y_pred)[0]
    paras=np.array([1.2,1,0.2,0,0,0,0,1,0.1,1.2,\
                        2,1,1.5,\
                        0.5,1,1,0.5,0.1,0.7,\
                        0.5,1,1,0.5,0.1,0.7,\
                        0.5,1,1,0.5,0.1,0.7,\
                        0.5,1,1,0.5,0.1,0.7])
    All_loss=[]
    for i in range(batch_size):
        # times=np.square(((y_pred[i]+0.6).astype(np.int32)-y_true[i]))
        # ---- Original 
        times=np.square(np.multiply((y_pred[i]-y_true[i]),paras))
        sums=times.sum()/37
        All_loss.append(sums)
    x_list=range(0,batch_size)    
    np.savetxt(myloss_save,All_loss,fmt='%f',delimiter=',')
    # --------- draw pictures 
    plt.figure()
    #plt.plot(x_list,All_loss,color='#2455C6')
    plt.plot(x_list,All_loss,color='#6495ED')
    plt.xlabel('时间/天',fontsize=13)
    plt.ylabel('WDD损失',fontsize=13)

    plt.savefig(figure_save)
    plt.show()
        # print(np.shape(loss))

def Calculate_deviations(files_test,label_test,save_path,loss_save,figure_save):
    x_test=np.loadtxt(files_test,delimiter=',')
    y_test=np.loadtxt(label_test,delimiter=',')
    # print (np.shape(x_test))
    # print(np.shape([x_test[0]]))
    # print(len(x_test))
    # exit (0)
    model=load_model(save_path)
    # exit(0)
    All_loss=[]
    x_list=range(0,len(x_test))
    for i in range (len(x_test)):
        x_small_test=np.reshape(x_test[i],(1,148))
        y_small_test=np.reshape(y_test[i],(1,37))
        loss,acc=model.evaluate(x=x_small_test,y=y_small_test,verbose=0)
        All_loss.append(loss)
    np.savetxt(loss_save,All_loss,fmt='%f',delimiter=',')
    # --------- draw pictures 
    plt.figure()
    plt.plot(x_list,All_loss,color='red')
    plt.xlabel('时间/天',fontsize=13)
    plt.ylabel('MSE损失',fontsize=13)

    plt.savefig(figure_save)
    plt.show()
    # exit(0)





if __name__ == "__main__":
    
    # user_sets={'EDB0714':29,'TNM0961':32,'HXL0968':33}

    # -------- run model for every user separately or it will report errors because of the cache ----------- 
    user_sets={'EDB0714':29}
    # user_sets={'TNM0961':32}
    # user_sets={'HXL0968':33}

    for username,length in user_sets.items():
        USERNAME=username 
        folder='Data/'+ USERNAME +'/Model/Feature/'
        save_path=folder+'model.h5'
        files_train='Data/'+USERNAME+'/feature/'+'data_train.csv'
        labels_train='Data/'+USERNAME+'/feature/'+'label_train.csv'
        files_test='Data/'+USERNAME+'/feature/'+'data_test.csv'
        predict_save='Data/'+USERNAME+'/feature/'+'predict.csv'
        label_test='Data/'+USERNAME+'/feature/'+'label_test.csv'
        loss_save='Data/'+USERNAME+'/feature/'+'loss.csv'
        figure_save='Data/'+USERNAME+'/feature/'+'loss7.jpg'
        figure_my_save='Data/'+USERNAME+'/feature/'+'myloss7.jpg'
        myloss_save='Data/'+USERNAME+'/feature/'+'myloss.csv'

        train(files_train,labels_train,save_path,files_test,predict_save)
        # retrain(files_train,labels_train,save_path,files_test,predict_save)

# ------------ calculate all deviations for all data(train+test)
        predict_save='Data/'+USERNAME+'/feature/'+'predict_all.csv'
        files_all='Data/'+USERNAME+'/feature/'+'data_all.csv'
        labels_all='Data/'+USERNAME+'/feature/'+'label_all.csv'
        loss_save='Data/'+USERNAME+'/feature/'+'loss_all.csv'
        figure_save='Data/'+USERNAME+'/feature/'+'loss_all9.jpg'
        figure_my_save='Data/'+USERNAME+'/feature/'+'myloss_all9.jpg'
        myloss_save='Data/'+USERNAME+'/feature/'+'myloss_all.csv'
        test(files_all,save_path,predict_save)
        Calculate_deviations(files_all,labels_all,save_path,loss_save,figure_save)
        my_loss_forFeatures(labels_all,predict_save,myloss_save,figure_my_save)

# ----------------------------------------------------------       

