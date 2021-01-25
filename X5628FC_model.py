# -*- coding:utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from keras import Sequential



##############################
#
# Model structure
#
##############################

def model_def():

    dropout_rate = 0.5 # Discard ratio
  
    model = Sequential()
  
    model.add(Conv2D(64, kernel_size = [24, 1], strides = [4, 1], padding = 'same', activation = 'relu', input_shape = ((20002, 5, 1))))
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
    model.add(Dropout(dropout_rate)) # Dropout 1
    '''
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(dropout_rate)) # Dropout 2
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(dropout_rate)) # Dropout 3
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(dropout_rate)) # Dropout 4
    '''
    model.add(Dense(2048, activation = 'relu')) # No Dropout on the last layer
    model.add(Dense(2, activation = 'softmax')) # Output normalization softmax
  
    return model



##############################
#
# Inspection area
#
##############################

if __name__ == '__main__':
  
    # Used to put test code here
    pass
