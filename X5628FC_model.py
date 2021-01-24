# -*- coding:utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from keras import Sequential



##############################
#
# 模型结构
#
##############################

def model_def():

    dropout_rate = 0.5 # 舍弃比率
  
    model = Sequential()
  
    model.add(Conv2D(64, kernel_size = [24, 1], strides = [4, 1], padding = 'same', activation = 'relu', input_shape = ((10001, 8, 1))))
    model.add(Conv2D(64, kernel_size = [24, 1], strides = [4, 1], padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = [24, 1], strides = [4, 1], padding = 'same', activation = 'relu'))
  
    model.add(MaxPooling2D(pool_size = (2, 1), strides = (2, 1)))
    model.add(BatchNormalization())
  
    model.add(Conv2D(128, kernel_size = [24, 1], strides = [4, 1], padding = 'same', activation = 'relu'))
    model.add(Conv2D(128, kernel_size = [24, 1], strides = [4, 1], padding = 'same', activation = 'relu'))
    model.add(Conv2D(128, kernel_size = [24, 1], strides = [4, 1], padding = 'same', activation = 'relu'))
  
    model.add(MaxPooling2D(pool_size = (2, 1), strides = (2, 1)))
  
    model.add(Flatten())
    model.add(BatchNormalization())
  
    model.add(Dense(2048, activation = 'relu'))
    model.add(Dropout(dropout_rate)) # 1
    
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(dropout_rate)) # 2
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(dropout_rate)) # 3
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(dropout_rate)) # 4
    
    model.add(Dense(8, activation = 'relu')) # 最后一层不加 Dropout
    model.add(Dense(2, activation = 'softmax')) # 输出归一化 softmax
  
    return model



##############################
#
# 检修区
#
##############################

if __name__ == '__main__':
  
    # 用来放测试代码
    pass
