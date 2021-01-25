# -*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Remove unnecessary information

# Garbage collection mechanism
import gc
gc.enable()

import numpy as np

# Print format
def fancy_print(n = None, c = None, s = '#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # Avoid confusion

# Set the GPU usage mode to be progressive to avoid full memory
# Get GPU list
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set the GPU to increase occupancy mode
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('Set the GPU to increase occupancy mode')
    except RuntimeError as e:
        # print error(no GPU mode)
        fancy_print('RuntimeError', e)



##############################
#
# Build iterator
#
##############################

from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rescale = 1. / 255) # Flip up and down, flip left and right
train_datagen = ImageDataGenerator(rescale = 1. / 255)
val_datagen = ImageDataGenerator(rescale = 1. / 255)

BATCH_SIZE = 32 # Size each time

train_generator = train_datagen.flow_from_directory(directory = './train/', target_size = (20002, 5),
                                                    color_mode = 'grayscale',
                                                    classes = ['pos', 'neg'],
                                                    #'categorical' will return 2D one-hot encoding labels,'binary' returns 1D binary labels, and'sparse' returns 1D integer labels
                                                    class_mode = 'categorical',
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True, # must shuffle
                                                    seed = 42)
val_generator = val_datagen.flow_from_directory(directory = './val/', target_size = (20002, 5),
                                                color_mode = 'grayscale',
                                                classes = ['pos', 'neg'],
                                                #'categorical' will return 2D one-hot encoding labels,'binary' returns 1D binary labels, and'sparse' returns 1D integer labels
                                                class_mode = 'categorical',
                                                batch_size = BATCH_SIZE,
                                                shuffle = True, # must shuffle
                                                seed = 42)



##############################
#
# Model building
#
##############################

# If the version is not compatible, then use these two lines of code, otherwise a warning will be reported and stop the porgram
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from sklearn import metrics
from keras.callbacks import ModelCheckpoint

# Import X5628FC_model.py
import X5628FC_model

clf = X5628FC_model.model_def()

clf.summary() # Print model structure

from keras.optimizers import Adam
clf.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr = 0.0001, decay = 0.00001),
            metrics = ['accuracy'])

'''
filename = 'best_model.h5'
modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
'''
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 10, restore_best_weights = True)

gc.collect() # Recycle all generations of garbage to avoid memory leaks




fancy_print('train_generator.next()[0]', train_generator.next()[0], '+')
fancy_print('train_generator.next()[1]', train_generator.next()[1], '+')
fancy_print('train_generator.next()[0].shape', train_generator.next()[0].shape, '+')
fancy_print('train_generator.next()[1].shape', train_generator.next()[1].shape, '+')

fancy_print('val_generator.next()[0]', val_generator.next()[0], '-')
fancy_print('val_generator.next()[1]', val_generator.next()[1], '-')
fancy_print('val_generator.next()[0].shape', val_generator.next()[0].shape, '-')
fancy_print('val_generator.next()[1].shape', val_generator.next()[1].shape, '-')

##############################
#
# Model training
#
##############################

# No need to count how many epochs, keras can count
history = clf.fit_generator(generator = train_generator,
                            epochs = 100,
                            validation_data = val_generator,

                            steps_per_epoch = int(6000 / BATCH_SIZE),
                            validation_steps = int(2000 / BATCH_SIZE),
                            initial_epoch = 0)

                            # callbacks = [early_stopping])

clf.save('best_model.h5')
