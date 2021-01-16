#-*- coding:utf-8 -*-

'''
本脚本用于读取数据
并进行onehot enconding处理
'''

import numpy as np

# 1开启debug模式，打印所有检查节点
# 0关闭debug模式，静默执行
debug_mode = 1

# 打印格式
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免了混乱
    
# 垃圾回收机制
import gc
gc.enable()



########################################
#
# 读取基因数据
#
########################################

# 采样点数量（一次全用会增加计算机负担）
smaple = 3000

# 读取所有基因序列
def read_data():

    print('-> reading anchor_data/seq.anchor1.pos.txt ...')
    # 第一组基因序列，正面
    f = open('anchor_data/seq.anchor1.pos.txt', 'r')
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
    if debug_mode:
        fancy_print('anchor1_pos.shape', np.array(anchor1_pos).shape, '+')

    f.close()
    gc.collect() # 回收全部代垃圾，避免内存泄露



    print('-> reading anchor_data/seq.anchor1.neg2.txt ...')
    # 第一组基因序列，负面
    f = open('anchor_data/seq.anchor1.neg2.txt', 'r')
    anchor1_neg2 = f.readlines()

    anchor1_neg2_temp = anchor1_neg2[:smaple] # 仅测试的时候使用这几句话，这里是深度复制，需要清理垃圾，自动无法清理
    del anchor1_neg2
    anchor1_neg2 = anchor1_neg2_temp
    del anchor1_neg2_temp

    # 替换为数字，并且去掉换行符号
    for num in range(len(anchor1_neg2)):
        anchor1_neg2[num] = anchor1_neg2[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
            .replace('C', 'C ').replace('N', 'N ').replace('\n', '')

    if debug_mode:
        fancy_print('anchor1_neg2.shape', np.array(anchor1_neg2).shape, '-')
    
    f.close()
    gc.collect() # 回收全部代垃圾，避免内存泄露



    print('-> reading anchor_data/seq.anchor2.pos.txt ...')
    # 第二组基因序列 正面
    f = open('anchor_data/seq.anchor2.pos.txt', 'r')
    anchor2_pos = f.readlines()

    anchor2_pos_temp = anchor2_pos[:smaple] # 仅测试的时候使用这几句话，这里是深度复制，需要清理垃圾，自动无法清理
    del anchor2_pos
    anchor2_pos = anchor2_pos_temp
    del anchor2_pos_temp

    # 替换为数字，并且去掉换行符号
    for num in range(len(anchor2_pos)):
        anchor2_pos[num] = anchor2_pos[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
            .replace('C', 'C ').replace('N', 'N ').replace('\n', '')

    if debug_mode:
        fancy_print('anchor2_pos.shape', np.array(anchor2_pos).shape, '+')

    f.close()
    gc.collect() # 回收全部代垃圾，避免内存泄露



    print('-> reading anchor_data/seq.anchor2.neg2.txt ...')
    # 第二组基因序列 负面
    f = open('anchor_data/seq.anchor2.neg2.txt', 'r')
    anchor2_neg2 = f.readlines()

    anchor2_neg2_temp = anchor2_neg2[:smaple] # 仅测试的时候使用这几句话，这里是深度复制，需要清理垃圾，自动无法清理
    del anchor2_neg2
    anchor2_neg2 = anchor2_neg2_temp
    del anchor2_neg2_temp

    # 替换为数字，并且去掉换行符号
    for num in range(len(anchor2_neg2)):
        anchor2_neg2[num] = anchor2_neg2[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
            .replace('C', 'C ').replace('N', 'N ').replace('\n', '')

    if debug_mode:
        fancy_print('anchor2_neg2.shape', np.array(anchor2_neg2).shape, '-')

    f.close()
    gc.collect() # 回收全部代垃圾，避免内存泄露

    '''
    # 验证用，打印前五项
    for line in anchor1_pos[0:5]:
        print(line)
    '''

    # 返回所有结果
    return anchor1_pos, anchor1_neg2, anchor2_pos, anchor2_neg2



########################################
#
# split
#
########################################

def split_dataset(anchor1_pos, anchor1_neg2, anchor2_pos, anchor2_neg2):

    split_rate = 0.9

    import math

    anchor1_pos_train = anchor1_pos[:math.floor(len(anchor1_pos)*0.9)]
    anchor1_pos_test = anchor1_pos[math.floor(len(anchor1_pos)*0.9):]

    anchor1_neg2_train = anchor1_neg2[:math.floor(len(anchor1_neg2)*0.9)]
    anchor1_neg2_test = anchor1_neg2[math.floor(len(anchor1_neg2)*0.9):]

    anchor2_pos_train = anchor2_pos[:math.floor(len(anchor2_pos)*0.9)]
    anchor2_pos_test = anchor2_pos[math.floor(len(anchor2_pos)*0.9):]

    anchor2_neg2_train = anchor2_neg2[:math.floor(len(anchor2_neg2)*0.9)]
    anchor2_neg2_test = anchor2_neg2[math.floor(len(anchor2_neg2)*0.9):]

    return anchor1_pos_train, anchor1_pos_test, \
           anchor1_neg2_train, anchor1_neg2_test, \
           anchor2_pos_train,anchor2_pos_test, \
           anchor2_neg2_train, anchor2_neg2_test



########################################
#
# 合并结果
#
########################################

def get_merged_result(anchor1_pos_train, anchor1_pos_test, \
                      anchor1_neg2_train, anchor1_neg2_test):

    # 训练数据
    # 生成结果数组(全1)
    train_pos_result = np.ones(len(anchor1_pos_train))
    # 生成结果数组(全0)
    train_neg2_result = np.zeros(len(anchor1_neg2_train))

    # 测试数据
    # 生成结果数组(全1)
    test_pos_result = np.ones(len(anchor1_pos_test))
    # 生成结果数组(全0)
    test_neg2_result = np.zeros(len(anchor1_neg2_test))
    
    # 合并预测结果
    # 二分类问题不需要onehot编码
    label_train = np.append(train_pos_result, train_neg2_result)
    # label_train = to_categorical(label_train) # 转换成onehot编码
    fancy_print('label_train.shape', label_train.shape, '*')

    label_test = np.append(test_pos_result, test_neg2_result)
    # label_test = to_categorical(label_test) # 转换成onehot编码
    fancy_print('label_test.shape', label_test.shape, '*')

    gc.collect() # 回收全部代垃圾，避免内存泄露

    return label_train, label_test



########################################
#
# onehot enconding
#
########################################

# 第一个参数是需要编码的数据，第二个参数是OneHotEncoder
def onehot_func(data, ATGC):
    
    data_onehot = []

    for i in data:
        # 把一维数组变成二维数组
        i = list(map(list, i.split()))
        data_onehot.append(np.transpose(ATGC.transform(i).toarray()))
    del data # 这里需要清理垃圾

    # 深度复制
    data_onehot_temp = np.array(data_onehot)
    del data_onehot
    data_onehot = data_onehot_temp
    del data_onehot_temp
    
    # 查看大小
    # fancy_print('data_onehot[0]', data_onehot[0], '+')
    if debug_mode:
        fancy_print('data_onehot.shape', data_onehot.shape, '+')
    gc.collect() # 回收全部代垃圾，避免内存泄露

    return data_onehot
    


def onehot_enconding(anchor1_pos_train, anchor1_pos_test, \
                     anchor1_neg2_train, anchor1_neg2_test, \
                     anchor2_pos_train, anchor2_pos_test, \
                     anchor2_neg2_train, anchor2_neg2_test):

    # 第一部分的正面数据
    # 转换成onehot编码
    from keras.utils.np_utils import to_categorical
    from sklearn import preprocessing

    # 默认的是handle_unknown='error'，即不认识的数据报错，改成ignore代表忽略，全部用0替代
    ATGC = preprocessing.OneHotEncoder(handle_unknown='ignore')
    # 打印一下输出数组，可以不打印
    fancy_print('one-hot enconding', '[[\'A\'],[\'T\'],[\'G\'],[\'C\']]\n' + str(ATGC.fit_transform([['A'],['T'],['G'],['C']]).toarray()))



    anchor1_pos_train_onehot = onehot_func(anchor1_pos_train, ATGC)
    anchor1_pos_test_onehot = onehot_func(anchor1_pos_test, ATGC)
    
    anchor1_neg2_train_onehot = onehot_func(anchor1_neg2_train, ATGC)
    anchor1_neg2_test_onehot = onehot_func(anchor1_neg2_test, ATGC)
    
    anchor2_pos_train_onehot = onehot_func(anchor2_pos_train, ATGC)
    anchor2_pos_test_onehot = onehot_func(anchor2_pos_test, ATGC)
    
    anchor2_neg2_train_onehot = onehot_func(anchor2_neg2_train, ATGC)
    anchor2_neg2_test_onehot = onehot_func(anchor2_neg2_test, ATGC)

    # 合并在一起
    train_onehot_1 = np.vstack((anchor1_pos_train_onehot, anchor1_neg2_train_onehot))
    train_onehot_2 = np.vstack((anchor2_pos_train_onehot, anchor2_neg2_train_onehot))

    test_onehot_1 = np.vstack((anchor1_pos_test_onehot, anchor1_neg2_test_onehot))
    test_onehot_2 = np.vstack((anchor2_pos_test_onehot, anchor2_neg2_test_onehot))



    return train_onehot_1, train_onehot_2, \
           test_onehot_1, test_onehot_2



########################################
#
# 扩充维度
#
########################################

def expand_dim(train_onehot_1, train_onehot_2, \
               test_onehot_1, test_onehot_2):

    # 为了CNN，扩充维度
    if debug_mode:
        fancy_print('train_onehot_1.shape', train_onehot_1.shape, '*')
    train_onehot_1 = train_onehot_1[:, :, :, np.newaxis]
    fancy_print('train_onehot_1.shape', train_onehot_1.shape, '*')

    if debug_mode:
        fancy_print('train_onehot_2.shape', train_onehot_2.shape, '*')
    train_onehot_2 = train_onehot_2[:, :, :, np.newaxis]
    fancy_print('train_onehot_2.shape', train_onehot_2.shape, '*')

    if debug_mode:
        fancy_print('test_onehot_1.shape', test_onehot_1.shape, '*')
    test_onehot_1 = test_onehot_1[:, :, :, np.newaxis]
    fancy_print('test_onehot_1.shape', test_onehot_1.shape, '*')

    if debug_mode:
        fancy_print('test_onehot_2.shape', test_onehot_2.shape, '*')
    test_onehot_2 = test_onehot_2[:, :, :, np.newaxis]
    fancy_print('test_onehot_2.shape', test_onehot_2.shape, '*')

    gc.collect() # 回收全部代垃圾，避免内存泄露

    return train_onehot_1, train_onehot_2, \
           test_onehot_1, test_onehot_2
    


########################################
#
# 检修区
#
########################################

if __name__ == '__main__':
    
    # 用来放测试代码
    pass
