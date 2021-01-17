#-*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 垃圾回收机制
import gc
gc.enable()

import numpy as np

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
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免了混乱



########################################
#
# 数据预处理部分
#
########################################

# 引入data_processing.py
import data_processing

train_onehot_1, train_onehot_2, \
val_onehot_1, val_onehot_2, \
test_onehot_1, test_onehot_2, \
label_train, label_val, label_test = data_processing.data_process()



########################################
#
# 深度学习部分
#
########################################

# 值1：不训练，直接读取h5文件
# 值0：训练，并生成h5文件
USE_EXIST_MODEL = 0

if USE_EXIST_MODEL == 0:

    # 如果出现版本不兼容，那么就用这两句代码，否则会报警告
    # import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()

    import numpy as np
    from sklearn import metrics

    from keras.callbacks import ModelCheckpoint

    import math
    # 导入model.py
    import model



    clf = model.model_def()
    gc.collect() # 回收全部代垃圾，避免内存泄露



    filename = 'best_model.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
    gc.collect() # 回收全部代垃圾，避免内存泄露



    clf.fit([train_onehot_1, train_onehot_2], label_train, validation_data=([val_onehot_1,val_onehot_2], label_val),
            epochs = 100, batch_size = 40,
            shuffle=True, callbacks = [modelCheckpoint])
    gc.collect() # 回收全部代垃圾，避免内存泄露

else:

    # 跳过训练，直接加载模型
    from keras.models import load_model
    clf = load_model('best_model.h5')











# 新加入内容，用来评估模型质量
# 计算auc和绘制roc_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

score = clf.evaluate([test_onehot_1, test_onehot_2], label_test)
print('loss & acc:', score)

# 打印所有内容
np.set_printoptions(threshold=np.inf)

# 利用model.predict获取测试集的预测概率
y_prob = clf.predict(x=[test_onehot_1, test_onehot_2])
fancy_print('y_prob', y_prob, '.')
fancy_print('y_prob.shape', y_prob.shape, '-')



# 为每个类别计算ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()



# 二分类问题
n_classes = label_test.shape[1] # n_classes = 2
fancy_print('n_classes', n_classes) # n_classes = 2

# 使用实际类别和预测概率绘制ROC曲线
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fancy_print('fpr', fpr)
fancy_print('tpr', tpr)
fancy_print('cnn_roc_auc', roc_auc)



plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

plt.show()
