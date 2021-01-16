from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from keras.layers import concatenate
from keras.layers import Input

from keras.models import Model

from keras.optimizers import Adam



# 打印格式
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免了混乱

# 1开启debug模式，打印所有检查节点
# 0关闭debug模式，静默执行
debug_mode = 1

########################################
#
# 模型结构
#
########################################

def model_def():

    sizex = 4 # 输入大小尺寸，四个碱基
    sizey = 10001 # 输入大小尺寸，基因片段长度
    dropout_rate = 0.1 # dropout比例

    # 全部使用Model模型
    # 第一部分模型
    # 输入层
    input_1 = Input(shape = (sizex, sizey, 1))
    # 1st Conv2D layer
    model_1 = Convolution2D(filters=32, kernel_size=[40, 4], padding='same')(input_1)
    model_1 = Activation('relu')(model_1)
    model_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model_1)
    # 2nd Conv2D layer
    model_1 = Convolution2D(filters=32, kernel_size=[40, 4], padding='same')(model_1)
    model_1 = Activation('relu')(model_1)
    model_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model_1)

    # 1st Fully connected Dense
    model_1 = Flatten()(model_1)
    model_1 = Dense(2048)(model_1); model_1 = Activation('relu')(model_1)
    model_1 = Dropout(dropout_rate)(model_1)
    # 2st Fully connected Dense
    model_1 = Dense(512)(model_1); model_1 = Activation('relu')(model_1)
    model_1 = Dropout(dropout_rate)(model_1)
    # 3st Fully connected Dense
    model_1 = Dense(128)(model_1); model_1 = Activation('relu')(model_1)





    # 第二部分模型
    # 输入层
    input_2 = Input(shape = (sizex, sizey, 1))
    # 1st Conv2D layer
    model_2 = Convolution2D(filters=32, kernel_size=[40, 4], padding='same')(input_2)
    model_2 = Activation('relu')(model_2)
    model_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model_2)
    # 2nd Conv2D layer
    model_2 = Convolution2D(filters=32, kernel_size=[40, 4], padding='same')(model_2)
    model_2 = Activation('relu')(model_2)
    model_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(model_2)

    # 1st Fully connected Dense
    model_2 = Flatten()(model_2)
    model_2 = Dense(2048)(model_2); model_2 = Activation('relu')(model_2)
    model_2 = Dropout(dropout_rate)(model_2)
    # 2st Fully connected Dense
    model_2 = Dense(512)(model_2); model_2 = Activation('relu')(model_2)
    model_2 = Dropout(dropout_rate)(model_2)
    # 3st Fully connected Dense
    model_2 = Dense(128)(model_2); model_2 = Activation('relu')(model_2)





    # 合并之后
    model_concat = concatenate([model_1, model_2], axis=-1)
    if debug_mode:
        fancy_print('model_concat.shape', model_concat.shape)
    # dense层
    model_concat = Dense(64, activation='relu')(model_concat)
    model_concat = Dropout(dropout_rate)(model_concat) # 避免过拟合
    model_concat = Dense(8, activation='relu')(model_concat)
    model_concat = Dropout(dropout_rate)(model_concat) # 避免过拟合
    # 使用softmax把输出值限制在01之间，但是效果比sigmoid好
    model_concat = Dense(1, activation='softmax')(model_concat)
    if debug_mode:
        fancy_print('model_concat.shape', model_concat.shape)
    
    model = Model(inputs = [input_1, input_2], outputs = model_concat)

    fancy_print('model.summary()', None, '*')
    model.summary()

    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(lr = 1e-4),
                  metrics = ['acc'])
    
    import gc
    gc.collect() # 回收全部代垃圾，避免内存泄露

    return model



########################################
#
# 检修区
#
########################################

if __name__ == '__main__':
    
    # 用来放测试代码
    pass
