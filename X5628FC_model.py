from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from keras import Sequential



##############################
#
# 模型结构
#
##############################

def model_def():

    sizex = 20002 # 输入大小尺寸，基因片段长度
    sizey = 4 # 输入大小尺寸，四个碱基
    dropout_rate = 0.2 # dropout比例，别弄太大，永远振荡，也别弄太小，否则过拟合

    

    CNN_height = 4
    CNN_width = 40

    model = Sequential()

    # height and width
    model.add(Conv2D(64,kernel_size=[CNN_height,CNN_width],strides=[4,4],padding='same',activation='relu',input_shape=(sizex,sizey,1)))
    model.add(Conv2D(64,kernel_size=[CNN_height,CNN_width],strides=[4,4],padding='same',activation='relu'))
    model.add(Conv2D(64,kernel_size=[CNN_height,CNN_width],strides=[4,4],padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(128,kernel_size=[CNN_height,CNN_width],strides=[4,4],padding='same',activation='relu'))
    model.add(Conv2D(128,kernel_size=[CNN_height,CNN_width],strides=[4,4],padding='same',activation='relu'))
    model.add(Conv2D(128,kernel_size=[CNN_height,CNN_width],strides=[4,4],padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(8, activation='relu')) # 最后一层别加 Dropout
    model.add(Dense(2, activation='softmax')) # softmax 似乎不太行

    return model



##############################
#
# 检修区
#
##############################

if __name__ == '__main__':
    
    # 用来放测试代码
    pass
