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

# 使用现有模型，禁止训练
USE_EXISTING_MODEL = 1

if USE_EXISTING_MODEL==0:
    
    ########################################
    #
    # 数据预处理部分
    #
    ########################################

    # 引入data_processing.py
    import data_processing

    # 读取数据
    anchor1_pos, anchor1_neg2, anchor2_pos, anchor2_neg2 = data_processing.read_data()

    # 分割数据集为训练集和测试集 0.9:0.1
    anchor1_pos_train, anchor1_pos_test, \
    anchor1_neg2_train, anchor1_neg2_test, \
    anchor2_pos_train,anchor2_pos_test, \
    anchor2_neg2_train, anchor2_neg2_test = \
    data_processing.split_dataset(anchor1_pos, anchor1_neg2, anchor2_pos, anchor2_neg2)

    # 获得标签结果
    label_train, label_test = \
    data_processing.get_merged_result(anchor1_pos_train, anchor1_pos_test, \
                                      anchor1_neg2_train, anchor1_neg2_test)

    # one-hot enconding
    train_onehot_1, train_onehot_2, \
    test_onehot_1, test_onehot_2 = \
    data_processing.onehot_enconding(anchor1_pos_train, anchor1_pos_test, \
                                     anchor1_neg2_train, anchor1_neg2_test, \
                                     anchor2_pos_train, anchor2_pos_test, \
                                     anchor2_neg2_train, anchor2_neg2_test)

    # 为了CNN扩展一个维度
    train_onehot_1, train_onehot_2, \
    test_onehot_1, test_onehot_2 = \
    data_processing.expand_dim(train_onehot_1, train_onehot_2, \
               test_onehot_1, test_onehot_2)



    # 用作训练的部分
    # test_onehot_1
    # test_onehot_2
    #
    # label_test



    ########################################
    #
    # 深度学习部分
    #
    ########################################

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
    modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_acc', save_best_only = True, mode = 'max')
    gc.collect() # 回收全部代垃圾，避免内存泄露



    clf.fit([train_onehot_1, train_onehot_2], label_train, epochs = 50, batch_size = 20, # 50 20
                  validation_split = 0.1, callbacks = [modelCheckpoint])
    gc.collect() # 回收全部代垃圾，避免内存泄露

else:

    # 加载模型
    from keras.models import load_model
    clf = load_model('best_model.h5')








# 用作训练的部分
# test_onehot_1
# test_onehot_2
#
# label_test



# 新加入内容，用来评估模型质量
# 计算auc和绘制roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from scipy import interp

import os
import matplotlib.pyplot as plt



# 利用model.predict获取测试集的预测值
y_score = clf.predict([test_onehot_1, test_onehot_2])
fancy_print('y_score', y_score, '.')

# onehot
y_temp_1 = []; y_temp_0 = []
for each in y_score: # 这个each是一个
    y_temp_1.append(float(each))
    y_temp_0.append(1-float(each))
y_prob = [y_temp_0, y_temp_1]; y_prob = np.transpose(y_prob)
fancy_print('y_prob', y_prob, '.')
fancy_print('y_prob.shape', y_prob.shape, '-')



# 利用model.predict_proba获取测试集的预测概率(0-1之间)
y_class = []
for i in y_score:
    if i >= 0.5:
        y_class.append(1)
    else:
        y_class.append(0)

# onehot
y_temp_1 = []; y_temp_0 = []
for each in y_class:
    if each == 1: y_temp_1.append(1); y_temp_0.append(0)
    else: y_temp_1.append(0); y_temp_0.append(1)
y_class = [y_temp_0, y_temp_1]; y_class = np.transpose(y_class)
fancy_print('y_class', y_class, '.')
fancy_print('y_class.shape', y_class.shape, '-')



# 为每个类别计算ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

# 二进制化输出
# 转成独热码
y_test = label_test.tolist()
y_temp_1 = []; y_temp_0 = []
for each in y_test:
    if each == 1: y_temp_1.append(1); y_temp_0.append(0)
    else: y_temp_1.append(0); y_temp_0.append(1)
y_test = [y_temp_0, y_temp_1]; y_test = np.transpose(y_test)
fancy_print('y_test', y_test, '.')
fancy_print('y_test.shape', y_test.shape, '-')



n_classes = y_test.shape[1] # n_classes = 2
fancy_print('n_classes', n_classes) # n_classes = 2

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fancy_print('fpr', fpr)
fancy_print('tpr', tpr)
fancy_print('CNN_roc_auc', roc_auc)



plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(ROC)')
plt.legend(loc="lower right")

plt.show()
