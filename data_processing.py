#-*- coding:utf-8 -*-

'''
本脚本用于读取数据
并进行onehot enconding处理
'''

import numpy as np

import gc # 垃圾回收机制
gc.enable()



# 1开启debug模式，打印所有检查节点
# 0关闭debug模式，静默执行
debug_mode = 1

# 采样点数量（一次全用会增加计算机负担）
smaple = 2000



# 打印格式
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免了混乱
    


########################################
#
# 读取基因数据
#
########################################

def read_data(name, file_dir):

    # 读取数据
    f = open(file_dir, 'r')
    data = f.readlines()

    data = data[:smaple] # 分割一个比较小的大小，用于测试

    # 替换为可以split的格式，并且去掉换行符号
    for num in range(len(data)):
        data[num] = data[num].replace('A', 'A ').replace('T', 'T ').replace('G', 'G ') \
                    .replace('C', 'C ').replace('N', 'N ').replace('\n', '')

    f.close()
        
    if debug_mode:
        fancy_print(name + '.shape', np.array(data).shape, '=')

    return data



########################################
#
# split
#
########################################

def data_split(data):

    val_split_rate = 0.1 # 0.1
    test_split_rate = 0.1 # 0.1
    train_split_rate = 1 - val_split_rate - test_split_rate # 0.8
    
    import math
    
    length = math.floor(len(data)) # 获取长度
    train = data[ : int(length * train_split_rate) ]
    val = data[ int(length * train_split_rate) :
                int(length * (train_split_rate + val_split_rate)) ]
    test = data[ int(length * (train_split_rate + val_split_rate)) : ]
    
    return train, val, test



########################################
#
# onehot enconding
#
########################################

# 第二个参数是需要编码的数据，第三个参数是OneHotEncoder
def onehot_func(name, data, ATGC):
    
    data_onehot = []

    for i in data:
        # 把一维数组变成二维数组
        i = list(map(list, i.split()))
        data_onehot.append(np.transpose(ATGC.transform(i).toarray()))

    data_onehot = np.array(data_onehot)
    
    if debug_mode:
        fancy_print(name + '.shape', data_onehot.shape, '+')

    return data_onehot





# 读取程序
def data_process():

    ########################################
    #
    # 读取基因数据
    #
    ########################################

    anchor1_pos = read_data('anchor1_pos', 'anchor_data/seq.anchor1.pos.txt')
    anchor1_neg2 = read_data('anchor1_neg2', 'anchor_data/seq.anchor1.neg2.txt')
    anchor2_pos = read_data('anchor2_pos', 'anchor_data/seq.anchor2.pos.txt')
    anchor2_neg2 = read_data('anchor2_neg2', 'anchor_data/seq.anchor2.neg2.txt')

    gc.collect() # 回收全部代垃圾，避免内存泄露
    

    
    ########################################
    #
    # split
    #
    ########################################
    
    anchor1_pos_train, anchor1_pos_val, anchor1_pos_test = data_split(anchor1_pos)
    anchor1_neg2_train, anchor1_neg2_val, anchor1_neg2_test = data_split(anchor1_neg2)

    anchor2_pos_train, anchor2_pos_val, anchor2_pos_test = data_split(anchor2_pos)
    anchor2_neg2_train, anchor2_neg2_val, anchor2_neg2_test = data_split(anchor2_neg2)

    gc.collect() # 回收全部代垃圾，避免内存泄露



    ########################################
    #
    # 生成结果
    #
    ########################################

    # onehot
    from keras.utils import to_categorical
    
    label_train = np.append(np.ones(len(anchor1_pos_train)), np.zeros(len(anchor1_neg2_train)))
    label_train = to_categorical(label_train) # 转换成onehot编码
    fancy_print('label_train', label_train, '*')

    label_val = np.append(np.ones(len(anchor1_pos_val)), np.zeros(len(anchor1_neg2_val)))
    label_val = to_categorical(label_val) # 转换成onehot编码
    fancy_print('label_val', label_val, '*')

    label_test = np.append(np.ones(len(anchor1_pos_test)), np.zeros(len(anchor1_neg2_test)))
    label_test = to_categorical(label_test) # 转换成onehot编码
    fancy_print('label_test', label_test, '*')

    gc.collect() # 回收全部代垃圾，避免内存泄露



    ########################################
    #
    # onehot enconding
    #
    ########################################

    # 转换成onehot编码
    from keras.utils import to_categorical
    from sklearn import preprocessing

    # 默认的是handle_unknown='error'，即不认识的数据报错，改成ignore代表忽略，全部用0替代
    ATGC = preprocessing.OneHotEncoder(handle_unknown='ignore')
    # 打印一下输出数组，可以不打印
    fancy_print('one-hot enconding', '[[\'A\'],[\'T\'],[\'G\'],[\'C\']]\n' + str(ATGC.fit_transform([['A'],['T'],['G'],['C']]).toarray()))



    anchor1_pos_train_onehot = onehot_func('anchor1_pos_train_onehot', anchor1_pos_train, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor1_neg2_train_onehot = onehot_func('anchor1_neg2_train_onehot', anchor1_neg2_train, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor2_pos_train_onehot = onehot_func('anchor2_pos_train_onehot', anchor2_pos_train, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor2_neg2_train_onehot = onehot_func('anchor2_neg2_train_onehot', anchor2_neg2_train, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    
    # 合并在一起
    train_onehot_1 = np.vstack((anchor1_pos_train_onehot, anchor1_neg2_train_onehot))
    train_onehot_2 = np.vstack((anchor2_pos_train_onehot, anchor2_neg2_train_onehot))
    gc.collect() # 回收全部代垃圾，避免内存泄露



    anchor1_pos_val_onehot = onehot_func('anchor1_pos_test_onehot', anchor1_pos_val, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor1_neg2_val_onehot = onehot_func('anchor1_neg2_test_onehot', anchor1_neg2_val, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor2_pos_val_onehot = onehot_func('anchor2_pos_test_onehot', anchor2_pos_val, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor2_neg2_val_onehot = onehot_func('anchor2_neg2_test_onehot', anchor2_neg2_val, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露    

    # 合并在一起
    test_onehot_1 = np.vstack((anchor1_pos_val_onehot, anchor1_neg2_val_onehot))
    test_onehot_2 = np.vstack((anchor2_pos_val_onehot, anchor2_neg2_val_onehot))
    gc.collect() # 回收全部代垃圾，避免内存泄露



    anchor1_pos_test_onehot = onehot_func('anchor1_pos_test_onehot', anchor1_pos_test, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor1_neg2_test_onehot = onehot_func('anchor1_neg2_test_onehot', anchor1_neg2_test, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor2_pos_test_onehot = onehot_func('anchor2_pos_test_onehot', anchor2_pos_test, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露
    anchor2_neg2_test_onehot = onehot_func('anchor2_neg2_test_onehot', anchor2_neg2_test, ATGC); gc.collect() # 回收全部代垃圾，避免内存泄露    

    # 合并在一起
    test_onehot_1 = np.vstack((anchor1_pos_test_onehot, anchor1_neg2_test_onehot))
    test_onehot_2 = np.vstack((anchor2_pos_test_onehot, anchor2_neg2_test_onehot))
    gc.collect() # 回收全部代垃圾，避免内存泄露



    ########################################
    #
    # 扩充维度
    #
    ########################################

    train_onehot_1 = train_onehot_1[:, :, :, np.newaxis]
    train_onehot_2 = train_onehot_2[:, :, :, np.newaxis]
    val_onehot_1 = test_onehot_1[:, :, :, np.newaxis]
    val_onehot_2 = test_onehot_2[:, :, :, np.newaxis]
    test_onehot_1 = test_onehot_1[:, :, :, np.newaxis]
    test_onehot_2 = test_onehot_2[:, :, :, np.newaxis]

    gc.collect() # 回收全部代垃圾，避免内存泄露



    return train_onehot_1, train_onehot_2, \
           val_onehot_1, val_onehot_2, \
           test_onehot_1, test_onehot_2, \
           label_train, label_val, label_test
           

    
########################################
#
# 检修区
#
########################################

if __name__ == '__main__':
    
    # 用来放测试代码
    pass
