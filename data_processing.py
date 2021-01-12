#-*- coding:utf-8 -*-

import numpy as np
import warnings
# 去除烦人的警告，都是些版本问题，无足轻重
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置GPU使用方式为渐进式，避免显存占满
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



# 垃圾回收机制
import gc
gc.enable()



# 采样点数量（一次全用会增加计算机负担）
smaple = -1



# 读取所有基因序列

# 第一组基因序列，正面
f = open('seq.anchor1.pos.txt', 'r')
anchor1_pos = f.readlines()

anchor1_pos_temp = anchor1_pos[:smaple] # 仅测试的时候使用这几句话，这里是深度复制，需要清理垃圾，自动无法清理
del anchor1_pos
anchor1_pos = anchor1_pos_temp
del anchor1_pos_temp

# 替换为数字，并且去掉换行符号
for num in range(len(anchor1_pos)):
    anchor1_pos[num] = anchor1_pos[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
        .replace('C', 'C ').replace('N', 'N ').replace('\n', '')
    
# fancy_print('anchor1_pos', anchor1_pos) # 应该全是数字
fancy_print('anchor1_pos.shape', np.array(anchor1_pos).shape, '+')

# 生成结果数组(全1)
anchor1_pos_result = np.ones(len(anchor1_pos))
f.close()

gc.collect() # 回收全部代垃圾，避免内存泄露



# 第一组基因序列，负面
f = open('seq.anchor1.neg2.txt', 'r')
anchor1_neg2 = f.readlines()

anchor1_neg2_temp = anchor1_neg2[:smaple] # 仅测试的时候使用这几句话，这里是深度复制，需要清理垃圾，自动无法清理
del anchor1_neg2
anchor1_neg2 = anchor1_neg2_temp
del anchor1_neg2_temp

# 替换为数字，并且去掉换行符号
for num in range(len(anchor1_neg2)):
    anchor1_neg2[num] = anchor1_neg2[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
        .replace('C', 'C ').replace('N', 'N ').replace('\n', '')
    
fancy_print('anchor1_neg2.shape', np.array(anchor1_neg2).shape, '-')
# 生成结果数组(全0)
anchor1_neg2_result = np.zeros(len(anchor1_neg2))
f.close()

gc.collect() # 回收全部代垃圾，避免内存泄露



# 第二组基因序列 正面
f = open('seq.anchor2.pos.txt', 'r')
anchor2_pos = f.readlines()

anchor2_pos_temp = anchor2_pos[:smaple] # 仅测试的时候使用这几句话，这里是深度复制，需要清理垃圾，自动无法清理
del anchor2_pos
anchor2_pos = anchor2_pos_temp
del anchor2_pos_temp

# 替换为数字，并且去掉换行符号
for num in range(len(anchor2_pos)):
    anchor2_pos[num] = anchor2_pos[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
        .replace('C', 'C ').replace('N', 'N ').replace('\n', '')

fancy_print('anchor2_pos.shape', np.array(anchor2_pos).shape, '+')
# 生成结果数组(全1)
anchor2_pos_result = np.ones(len(anchor2_pos))
f.close()

gc.collect() # 回收全部代垃圾，避免内存泄露



# 第二组基因序列 负面
f = open('seq.anchor2.neg2.txt', 'r')
anchor2_neg2 = f.readlines()

anchor2_neg2_temp = anchor2_neg2[:smaple] # 仅测试的时候使用这几句话，这里是深度复制，需要清理垃圾，自动无法清理
del anchor2_neg2
anchor2_neg2 = anchor2_neg2_temp
del anchor2_neg2_temp

# 替换为数字，并且去掉换行符号
for num in range(len(anchor2_neg2)):
    anchor2_neg2[num] = anchor2_neg2[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
        .replace('C', 'C ').replace('N', 'N ').replace('\n', '')

fancy_print('anchor2_neg2.shape', np.array(anchor2_neg2).shape, '-')
# 生成结果数组(全0)
anchor2_neg2_result = np.zeros(len(anchor2_neg2))
f.close()

gc.collect() # 回收全部代垃圾，避免内存泄露



'''
# 验证用，打印前五项
for line in anchor1_pos[0:5]:
    print(line)
'''



# 如果出现版本不兼容，那么就用这两句代码，否则会报警告
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
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import math

from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from keras.layers import concatenate
from keras.models import Sequential
from keras.models import Model



def model_def():

    sizex = 4 # 输入大小尺寸，四个碱基
    sizey = 10001 # 输入大小尺寸，基因片段长度
    dropout_rate = 0.2 # dropout比例

    # 第一部分模型
    model_1 = Sequential()
    # 1st Conv2D layer
    model_1.add(Convolution2D(filters=32, kernel_size=[40, 4], padding='same', input_shape=(sizex, sizey, 1)))
    model_1.add(Activation('relu'))
    model_1.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    # 2nd Conv2D layer
    model_1.add(Convolution2D(filters=32, kernel_size=[40, 4], padding='same'))
    model_1.add(Activation('relu'))
    model_1.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # 1st Fully connected Dense
    model_1.add(Flatten());
    model_1.add(Dense(2048)); model_1.add(Activation('relu'))
    model_1.add(Dropout(dropout_rate))
    # 2st Fully connected Dense
    model_1.add(Dense(512)); model_1.add(Activation('relu'))
    model_1.add(Dropout(dropout_rate))
    # 3st Fully connected Dense
    model_1.add(Dense(128)); model_1.add(Activation('relu'))
    model_1.add(Dropout(dropout_rate))
    # 4st Fully connected Dense
    model_1.add(Dense(16)); model_1.add(Activation('relu'))
    model_1.add(Dropout(dropout_rate))
    # 5nd Fully connected Dense
    model_1.add(Dense(1)); model_1.add(Activation('relu'))



    # 第二部分模型
    model_2 = Sequential()
    # 1st Conv2D layer
    model_2.add(Convolution2D(filters=32, kernel_size=[40, 4], padding='same', input_shape=(sizex, sizey, 1)))
    model_2.add(Activation('relu'))
    model_2.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    # 2nd Conv2D layer
    model_2.add(Convolution2D(filters=32, kernel_size=[40, 4], padding='same'))
    model_2.add(Activation('relu'))
    model_2.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # 1st Fully connected Dense
    model_2.add(Flatten());
    model_2.add(Dense(2048)); model_2.add(Activation('relu'))
    model_2.add(Dropout(dropout_rate))
    # 2st Fully connected Dense
    model_2.add(Dense(512)); model_2.add(Activation('relu'))
    model_2.add(Dropout(dropout_rate))
    # 3st Fully connected Dense
    model_2.add(Dense(128)); model_2.add(Activation('relu'))
    model_2.add(Dropout(dropout_rate))
    # 4st Fully connected Dense
    model_2.add(Dense(16)); model_2.add(Activation('relu'))
    model_2.add(Dropout(dropout_rate))
    # 5nd Fully connected Dense
    model_2.add(Dense(1));
    model_2.add(Activation('relu'))



    # 合并之后
    model = Sequential()
    model_concat = concatenate([model_1.output, model_2.output], axis=-1)
    fancy_print('model_concat.shape', model_concat.shape)
    # 不需要这一层了
    '''
    model_concat = Dense(512, activation='relu')(model_concat)
    model_concat = Dense(64, activation='relu')(model_concat)
    model_concat = Dense(2, activation='relu')(model_concat)
    fancy_print('model_concat.shape', model_concat.shape)
    '''
    model = Model(inputs = [model_1.input, model_2.input], outputs = model_concat)

    fancy_print('model_1.summary()', model_1.summary(), '=')
    fancy_print('model_2.summary()', model_2.summary(), '=')
    fancy_print('model.summary()', model.summary(), '=')

    return model





# 第一部分的正面数据
# 转换成onehot编码
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing

# 默认的是handle_unknown='error'，即不认识的数据报错，改成ignore代表忽略，全部用0替代
ATGC = preprocessing.OneHotEncoder(handle_unknown='ignore')
# 打印一下输出数组，可以不打印
fancy_print('one-hot enconding', ATGC.fit_transform([['A'],['T'],['G'],['C']]).toarray())



anchor1_pos_onehot = []
# 第一部分的正面数据
for i in anchor1_pos:
    # 把一维数组变成二维数组
    i = list(map(list, i.split()))
    anchor1_pos_onehot.append(np.transpose(ATGC.transform(i).toarray()))
del anchor1_pos # 这里需要清理垃圾
anchor1_pos_onehot = np.array(anchor1_pos_onehot)
# 查看大小
fancy_print('anchor1_pos_onehot[0]', anchor1_pos_onehot[0], '+')
fancy_print('anchor1_pos_onehot.shape', anchor1_pos_onehot.shape, '+')
gc.collect() # 回收全部代垃圾，避免内存泄露



anchor1_neg2_onehot = []
# 第一部分的负面数据
for i in anchor1_neg2:
    i = list(map(list, i.split()))
    anchor1_neg2_onehot.append(np.transpose(ATGC.transform(i).toarray()))
del anchor1_neg2 # 这里需要清理垃圾
anchor1_neg2_onehot = np.array(anchor1_neg2_onehot)
# 查看大小
fancy_print('anchor1_neg2_onehot[0]', anchor1_neg2_onehot[0], '-')
fancy_print('anchor1_neg2_onehot.shape', anchor1_neg2_onehot.shape, '-')
gc.collect() # 回收全部代垃圾，避免内存泄露



anchor2_pos_onehot = []
# 第二部分的正面数据
for i in anchor2_pos:
    i = list(map(list, i.split()))
    anchor2_pos_onehot.append(np.transpose(ATGC.transform(i).toarray()))
del anchor2_pos # 这里需要清理垃圾
anchor2_pos_onehot = np.array(anchor2_pos_onehot)
# 查看大小
fancy_print('anchor2_pos_onehot[0]', anchor2_pos_onehot[0], '+')
fancy_print('anchor2_pos_onehot.shape', anchor2_pos_onehot.shape, '+')
gc.collect() # 回收全部代垃圾，避免内存泄露



anchor2_neg2_onehot = []
# 第二部分的负面数据
for i in anchor2_neg2:
    i = list(map(list, i.split()))
    anchor2_neg2_onehot.append(np.transpose(ATGC.transform(i).toarray()))
del anchor2_neg2 # 这里需要清理垃圾
anchor2_neg2_onehot = np.array(anchor2_neg2_onehot)
# 查看大小
fancy_print('anchor2_neg2_onehot[0]', anchor2_neg2_onehot[0], '-')
fancy_print('anchor2_neg2_onehot.shape', anchor2_neg2_onehot.shape, '-')
gc.collect() # 回收全部代垃圾，避免内存泄露





# 为了CNN，扩充维度
training_data_1 = np.vstack((anchor1_pos_onehot, anchor1_neg2_onehot))
fancy_print('training_data_1.shape', training_data_1.shape, '*')
training_data_1 = training_data_1[:, :, :, np.newaxis]
fancy_print('training_data_1.shape', training_data_1.shape, '*')

training_data_2 = np.vstack((anchor2_pos_onehot, anchor2_neg2_onehot))
fancy_print('training_data_2.shape', training_data_2.shape, '*')
training_data_2 = training_data_2[:, :, :, np.newaxis]
fancy_print('training_data_2.shape', training_data_2.shape, '*')

gc.collect() # 回收全部代垃圾，避免内存泄露

# 合并预测结果
# 二分类问题不需要onehot编码
label_1 = np.append(anchor1_pos_result, anchor1_neg2_result)
# label_1 = to_categorical(label_1) # 转换成onehot编码
fancy_print('label_1.shape', label_1.shape)

label_2 = np.append(anchor2_pos_result, anchor2_neg2_result)
# label_2 = to_categorical(label_2) # 转换成onehot编码
fancy_print('label_2.shape', label_2.shape)

# 合并label1和2
label = []
for i in range(len(label_1)):
    label.append([label_1[i], label_2[i]])

label_temp = np.array(label)
del label
label = label_temp
del label_temp

gc.collect() # 回收全部代垃圾，避免内存泄露





model = model_def()
gc.collect() # 回收全部代垃圾，避免内存泄露



model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 0.0001),
                  metrics = ['acc'])
gc.collect() # 回收全部代垃圾，避免内存泄露

filename = 'best_model.h5'
modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_acc', save_best_only = True, mode = 'max')
gc.collect() # 回收全部代垃圾，避免内存泄露



model.fit([training_data_1, training_data_2], label, epochs = 500, batch_size = 20,
              validation_split = 0.1, callbacks = [modelCheckpoint])
gc.collect() # 回收全部代垃圾，避免内存泄露
