#-*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 去除不必要的信息

# 垃圾回收机制
import gc
gc.enable()

import numpy as np

# 打印格式
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免了混乱

# 设置GPU使用方式为渐进式，避免显存占满
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('设置GPU为增长式占用')
    except RuntimeError as e:
        # 打印异常
        fancy_print('RuntimeError', e)



##############################
#
# 构建迭代器
#
##############################

from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=1. / 255) # 上下翻转 左右翻转
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory='train/', target_size=(20002,4),
                                                    color_mode='grayscale',
                                                    classes=['pos','neg'],
                                                    class_mode='categorical', # "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                    batch_size=32, shuffle=True)

val_generator = val_datagen.flow_from_directory(directory='val/', target_size=(20002,4),
                                                color_mode='grayscale',
                                                classes=['pos','neg'],
                                                class_mode='categorical', # "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                batch_size=32, shuffle = False) # 不要乱



##############################
#
# 模型搭建
#
##############################

# 如果出现版本不兼容，那么就用这两句代码，否则会报警告
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from sklearn import metrics
from keras.callbacks import ModelCheckpoint

# 导入X5628FC_model.py
import X5628FC_model

clf = X5628FC_model.model_def()

clf.summary() # 打印模型结构

from keras.optimizers import Adam
clf.compile(loss = 'categorical_crossentropy', # binary_crossentropy
            optimizer = Adam(lr=0.0001,decay=0.00001),
            metrics = ['accuracy'])

filename = 'best_model.h5'
modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max')

print('-> 模型搭建成功！')
print()

gc.collect() # 回收全部代垃圾，避免内存泄露

##############################
#
# 模型训练
#
##############################

clf.fit_generator(generator=train_generator,
                  epochs=50,
                  validation_data=val_generator,
                  # validation_steps=2000,
                  callbacks=[modelCheckpoint]) # class_weight='auto',
