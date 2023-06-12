import linecache,random


def count_line(files_in):
    file_open=open(files_in,'r')
    count=0
    for line in file_open:
        count+=1
    file_open.close()
    return count   

def Random_Num(index,number):
    '''
    Generate a list of index from index randomly
     index : the index of files
     number : the length of list generated
     return : a list of index  
    '''
    return random.sample(list(index),number)

def train_test_generate(data_all,label_all,train_data_save,train_label_save,test_data_save,test_label_save,rate=0.8):
    '''
    we extract 'rate' (the defaults is 0.8) percent data for training, and the rest for testing. 
    
    '''
    
    data_in=open(data_all,'r')
    label_in=open(label_all,'r')
    data_train=open(train_data_save,'wt')
    data_test=open(test_data_save,'wt')
    label_train=open(train_label_save,'wt')
    label_test=open(test_label_save,'wt')
    data_num=count_line(data_all)
    train_num=data_num*rate
    # print(train_num)
    index=0
    for line in data_in:
        if index<train_num:
            data_train.writelines(line)
        else:
            data_test.writelines(line)
        index=index+1

    data_train.close()
    data_test.close()

    index=0
    for label in label_in:
        if index<train_num:
            label_train.writelines(label)
        else:
            label_test.writelines(label)
        index=index+1
    label_train.close()
    label_test.close()
    data_in.close()
    label_in.close()


if __name__ == "__main__":
    working_folder='Data/Mix/'
    # ----- disorder all mixed data ------
    file_open='Data/Mix/'+'Mix_all_loss.csv'
    label_open='Data/Mix/'+'Mix_all_label.csv'
    file_save=open('Data/Mix/'+'Disorder_loss.csv','wt')
    label_save=open('Data/Mix/'+'Disorder_label.csv','wt')
    line_counts=count_line(label_open)
    index=[n for n in range(0,line_counts)] 
    Random_index=Random_Num(index,line_counts)
    # print(Random_index)
    for num in Random_index:
        line=linecache.getline(file_open,num+1)
        label=linecache.getline(label_open,num+1)
        file_save.writelines(line)
        label_save.writelines(label)
    file_save.close()
    label_save.close()
    # --------- generate tran and test data for MLP
    data_all='Data/Mix/'+'Disorder_loss.csv'
    label_all='Data/Mix/'+'Disorder_label.csv'
    data_train='Data/Mix/'+'train.csv'
    label_train='Data/Mix/'+'label_train.csv'
    data_test='Data/Mix/'+'test.csv'
    label_test='Data/Mix/'+'label_test.csv'
    train_test_generate(data_all,label_all,data_train,label_train,data_test,label_test,rate=0.7)
    print('ok')












