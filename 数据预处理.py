import numpy as np
import warnings
# 去除烦人的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置GPU使用方式
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)



# 打印格式
def fancy_print(n=None, c=None, s='#'):
    print(s*40);
    print(n);
    print(c);
    print(s*40);


    
# 读取所有基因序列

# 第一组基因序列，正面
f = open('seq.anchor1.pos.txt', 'r')
anchor1_pos = f.readlines()
anchor1_pos = anchor1_pos[:10000] # 仅测试的时候使用这句话

# 替换为数字，并且去掉换行符号
for num in range(len(anchor1_pos)):
    anchor1_pos[num] = anchor1_pos[num].replace('A', '0 ');
    anchor1_pos[num] = anchor1_pos[num].replace('T', '1 ');
    anchor1_pos[num] = anchor1_pos[num].replace('G', '2 ');
    anchor1_pos[num] = anchor1_pos[num].replace('C', '3 ');
    anchor1_pos[num] = anchor1_pos[num].replace('N', 'N ');
    anchor1_pos[num] = anchor1_pos[num].replace('\n', '');
    
# fancy_print('anchor1_pos', anchor1_pos) # 应该全是数字
fancy_print('anchor1_pos.shape', np.array(anchor1_pos).shape, '+')
'''
# 测试的时候取前一百项，这样快一点
fancy_print('anchor1_pos.shape', anchor1_pos.shape)
fancy_print('anchor1_pos[:100].shape', anchor1_pos[:100].shape)
'''
# 生成结果数组(全1)
anchor1_pos_result = np.ones(len(anchor1_pos))
f.close()



# 第一组基因序列，负面
f = open('seq.anchor1.neg2.txt', 'r')
anchor1_neg2 = f.readlines()
anchor1_neg2 = anchor1_neg2[:10000] # 仅测试的时候使用这句话

# 替换为数字，并且去掉换行符号
for num in range(len(anchor1_neg2)):
    anchor1_neg2[num] = anchor1_neg2[num].replace('A', '0 ');
    anchor1_neg2[num] = anchor1_neg2[num].replace('T', '1 ');
    anchor1_neg2[num] = anchor1_neg2[num].replace('G', '2 ');
    anchor1_neg2[num] = anchor1_neg2[num].replace('C', '3 ');
    anchor1_neg2[num] = anchor1_neg2[num].replace('N', 'N ');
    anchor1_neg2[num] = anchor1_neg2[num].replace('\n', '');
    
fancy_print('anchor1_neg2.shape', np.array(anchor1_neg2).shape, '-')
# 生成结果数组(全0)
anchor1_neg2_result = np.zeros(len(anchor1_neg2))
f.close()
'''
# 第二组基因序列
f = open('seq.anchor2.pos.txt', 'r')
anchor2_pos = f.readlines()
f.close()

f = open('seq.anchor2.neg2.txt', 'r')
anchor2_neg2 = f.readlines()
f.close()
'''

'''
# 验证用，打印前五项
for line in anchor1_pos[0:5]:
    print(line)
'''



# 如果出现版本不兼容，那么就用这两句代码
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()



import sys
import os, re
import random
import datetime
import numpy as np
import hickle as hkl
from sklearn import metrics



from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint



import math

def model_def():

    sizex = 4
    sizey = 10001

    model = Sequential()



    # 1st Conv2D layer
    model.add(Convolution2D(filters=32,kernel_size=[4, 4],padding='same',input_shape=(sizex, sizey, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding="same"))
     
    # 2nd Conv2D layer
    model.add(Convolution2D(filters=32,kernel_size=(4, 4),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding="same"))
     
    # 1st Fully connected Dense
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    # 2st Fully connected Dense
    model.add(Dense(128))
    model.add(Activation('relu'))

    # 3st Fully connected Dense
    model.add(Dense(16))
    model.add(Activation('relu'))
     
    # 4nd Fully connected Dense
    model.add(Dense(2))
    model.add(Activation('softmax'))



    print(model.summary())

    return model




# 第一部分的正面数据
# 转换成onehot编码
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing

# 默认的是handle_unknown='error'，即不认识的数据报错，改成ignore代表忽略
ATGC = preprocessing.OneHotEncoder(handle_unknown='ignore') 
fancy_print('one-hot enconding', ATGC.fit_transform([['0'],['1'],['2'],['3']]).toarray())

anchor1_pos_onehot = []
for i in anchor1_pos:
    # fancy_print('np.transpose(to_categorical(i.split()))', np.transpose(to_categorical(i.split())), '*')
    # 把一维数组变成二维数组
    i = list(map(list, i.split()))
    anchor1_pos_onehot.append(np.transpose(ATGC.transform(i).toarray()))
del anchor1_pos
anchor1_pos_onehot = np.array(anchor1_pos_onehot)
# 查看大小
fancy_print('anchor1_pos_onehot[0]', anchor1_pos_onehot[0], '+')
fancy_print('anchor1_pos_onehot.shape', anchor1_pos_onehot.shape, '+')



anchor1_neg2_onehot = []
# 第一部分的负面数据
for i in anchor1_neg2:
    i = list(map(list, i.split()))
    anchor1_neg2_onehot.append(np.transpose(ATGC.transform(i).toarray()))
del anchor1_neg2
anchor1_neg2_onehot = np.array(anchor1_neg2_onehot)
# 查看大小
fancy_print('anchor1_neg2_onehot[0]', anchor1_neg2_onehot[0], '-')
fancy_print('anchor1_neg2_onehot.shape', anchor1_neg2_onehot.shape, '-')










training_data = np.vstack((anchor1_pos_onehot, anchor1_neg2_onehot))
fancy_print('training_data.shape', training_data.shape)
training_data = training_data[:, :, :, np.newaxis]
fancy_print('training_data.shape', training_data.shape)

# 合并预测结果

label = np.append(anchor1_pos_result, anchor1_neg2_result)
label = to_categorical(label) # 转换成onehot编码
fancy_print('label.shape', label.shape)



model = model_def()


'''
model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizers.Adam(lr = 0.00001),
                  metrics = ['acc', f1])
'''

model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 0.00001),
                  metrics = ['acc'])

filename = 'best_model.h5'
modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_acc', save_best_only = True, mode = 'max')



model.fit(training_data, label, epochs = 100, batch_size = 50,
              validation_split = 0.1, callbacks = [modelCheckpoint])














