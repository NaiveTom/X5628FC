# -*- coding:utf-8 -*-

'''
本脚用于生成 png 文件
'''

import numpy as np
import os

import gc # 垃圾回收机制
gc.enable()



# 采样点数量（一次全用会增加计算机负担）
# 全部使用：None
smaple = None # 测试成功，全部使用



# 打印格式
def fancy_print(n=None, c=None, s='#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免了混乱



##############################
#
# 读取基因数据，并加上空格，方便之后处理
#
##############################

def read_data(name, file_dir): # name是用于打印的

    # 读取数据
    f = open(file_dir, 'r')
    data = f.readlines()

    data = data[ : smaple ] # 分割一个比较小的大小，用于测试，如果是None， 那么就选取全部

    # 替换为可以split的格式，并且去掉换行符号
    for num in range( len(data) ):
        data[num] = data[num].replace('A', '0 ').replace('C', '1 ').replace('G', '2 ') \
                    .replace('T', '3 ').replace('N', 'N ').replace('\n', '') # N不编码

    f.close()
        
    fancy_print(name + '.shape', np.array(data).shape, '=')

    return data



##############################
#
# 分割数据集变成测试集、验证集、训练集
#
##############################

def data_split(name, data):

    val_split_rate = 0.1 # 0.1
    test_split_rate = 0.1 # 0.1
    train_split_rate = 1 - val_split_rate - test_split_rate # 0.8

    print('-' * 40); print(name); print()

    print('train_split_rate', train_split_rate)
    print('val_split_rate', val_split_rate)
    print('test_split_rate', test_split_rate)
    
    print()
    
    import math
    
    length = math.floor(len(data)) # 获取长度
    train = data[ : int(length * train_split_rate) ]
    val = data[ int(length * train_split_rate) :
                int(length * (train_split_rate + val_split_rate)) ]
    test = data[ int(length * (train_split_rate + val_split_rate)) : ]

    print('len(train_set)', len(train))
    print('len(val_set)', len(val))
    print('len(test_set)', len(test))
    
    print('-' * 40); print()
    
    return train, val, test



##############################
#
# onehot enconding
#
##############################

# 第二个参数是需要编码的数据，第三个参数是OneHotEncoder
def onehot_func(data, ATGC):
    
    data_onehot = []

    for i in data:
        # 把一维数组变成二维数组
        i = list(map(list, i.split()))
        data_onehot.append(np.transpose(ATGC.transform(i).toarray()))

    data_onehot = np.array(data_onehot)

    return data_onehot





##############################
#
# 调用函数进行信息处理
#
##############################

def data_process():

    ####################
    # 读取基因数据
    ####################

    anchor1_pos = read_data('anchor1_pos', 'anchor_data/seq.anchor1.pos.txt')
    anchor1_neg2 = read_data('anchor1_neg2', 'anchor_data/seq.anchor1.neg2.txt')
    anchor2_pos = read_data('anchor2_pos', 'anchor_data/seq.anchor2.pos.txt')
    anchor2_neg2 = read_data('anchor2_neg2', 'anchor_data/seq.anchor2.neg2.txt')

    gc.collect()  # 回收全部代垃圾，避免内存泄露



    ####################
    # 调用函数进行分割处理
    ####################

    anchor1_pos_train, anchor1_pos_val, anchor1_pos_test = data_split('anchor1_pos', anchor1_pos)
    anchor1_neg2_train, anchor1_neg2_val, anchor1_neg2_test = data_split('anchor1_neg2', anchor1_neg2)
    anchor2_pos_train, anchor2_pos_val, anchor2_pos_test = data_split('anchor2_pos', anchor2_pos)
    anchor2_neg2_train, anchor2_neg2_val, anchor2_neg2_test = data_split('anchor2_neg2', anchor2_neg2)

    gc.collect()  # 回收全部代垃圾，避免内存泄露





    ##############################
    #
    # 写入h5文件（废弃）写入图片
    #
    ##############################

    # 写入图片
    # pip install imageio
    import imageio
    from skimage import img_as_ubyte

    # 转换成onehot编码
    from keras.utils import to_categorical
    from sklearn import preprocessing

    # 默认的是handle_unknown='error'，即不认识的数据报错，改成ignore代表忽略，全部用0替代
    ATGC = preprocessing.OneHotEncoder( handle_unknown = 'ignore' )
    # 打印一下输出数组，可以不打印
    fancy_print('one-hot enconding',
                '[ [A], [C], [G], [T] ]\n' + str(ATGC.fit_transform([['0'], ['1'], ['2'], ['3']]).toarray()))

    ####################
    # 训练集存成图片
    ####################

    LEN_PER_LOAD = 1000 # 越小越快，1000正好

    pic_num = 0

    for i in range( int(len(anchor1_pos_train)/LEN_PER_LOAD)+1 ): # 这里分块处理，一次处理1000个，因为onehot是二阶复杂度，别弄太大

        # 显示一下百分比
        print('\n正在分块存储训练集和标签，块编号 =', str(i), '/', int( len(anchor1_pos_train)/LEN_PER_LOAD) )

        if (i+1)*LEN_PER_LOAD > len(anchor1_pos_train): # 这段代码处理小尾巴问题

            try: # 有可能小尾巴是0
                anchor1_pos_train_onehot = onehot_func(anchor1_pos_train[ i*LEN_PER_LOAD : ], ATGC)
                anchor1_neg2_train_onehot = onehot_func(anchor1_neg2_train[ i*LEN_PER_LOAD : ], ATGC)
                anchor2_pos_train_onehot = onehot_func(anchor2_pos_train[ i*LEN_PER_LOAD : ], ATGC)
                anchor2_neg2_train_onehot = onehot_func(anchor2_neg2_train[ i*LEN_PER_LOAD : ], ATGC)
            except:
                print('最后一个块大小为0，已跳过本块（避免报错）')

        else: # 这段代码处理正常分块问题
            
            anchor1_pos_train_onehot = onehot_func(anchor1_pos_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ATGC)
            anchor1_neg2_train_onehot = onehot_func(anchor1_neg2_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ATGC)
            anchor2_pos_train_onehot = onehot_func(anchor2_pos_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ATGC)
            anchor2_neg2_train_onehot = onehot_func(anchor2_neg2_train[ i*LEN_PER_LOAD : (i+1)*LEN_PER_LOAD ], ATGC)

        # print('单块大小', anchor1_pos_train_onehot.shape)

        # 合并在一起
        train_pos = np.hstack((anchor1_pos_train_onehot, anchor2_pos_train_onehot)) # 积极&积极 横着合并 dstack 竖着合并 hstack
        train_neg = np.hstack((anchor1_neg2_train_onehot, anchor2_neg2_train_onehot)) # 消极&消极 横着合并 dstack 竖着合并 hstack


        print('合块大小', train_pos.shape)
        print('正在生成PNG ...')



        if train_pos.shape[0]==0 or train_pos.shape[1]==0 or train_pos.shape[2]==0:
            print('无效空块，已跳过！')
            continue # 空块，跳过循环



        # pip install tqdm
        # 进度条
        import tqdm

        # 写入一张张图片
        for j in tqdm.trange( len(train_pos), ascii=True ):
            imageio.imwrite('train/pos/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(train_pos[j]))) # 必须转置，因为PNG是反的
            imageio.imwrite('train/neg/'+str(pic_num)+'.png', img_as_ubyte(np.transpose(train_neg[j]))) # 必须转置，因为PNG是反的
            pic_num += 1
        
        gc.collect()  # 回收全部代垃圾，避免内存泄露



    ####################
    # 验证集存成图片
    ####################

    print('\n\n正在把验证集和标签写入 png 中 ...')

    anchor1_pos_val_onehot = onehot_func(anchor1_pos_val, ATGC)
    anchor1_neg2_val_onehot = onehot_func(anchor1_neg2_val, ATGC)
    anchor2_pos_val_onehot = onehot_func(anchor2_pos_val, ATGC)
    anchor2_neg2_val_onehot = onehot_func(anchor2_neg2_val, ATGC)

    # print('单块大小', anchor1_pos_val_onehot.shape)

    # 合并在一起
    val_pos = np.hstack((anchor1_pos_val_onehot, anchor2_pos_val_onehot)) # 积极&积极 横着合并 dstack 竖着合并 hstack
    val_neg = np.hstack((anchor1_neg2_val_onehot, anchor2_neg2_val_onehot)) # 消极&消极 横着合并 dstack 竖着合并 hstack

    print('合块大小', val_pos.shape)
    print('正在生成PNG ...')

    # 写入一张张图片
    for j in tqdm.trange( len(val_pos), ascii=True ):
        imageio.imwrite('val/pos/'+str(j)+'.png', img_as_ubyte(np.transpose(val_pos[j]))) # 必须转置，因为PNG是反的
        imageio.imwrite('val/neg/'+str(j)+'.png', img_as_ubyte(np.transpose(val_neg[j]))) # 必须转置，因为PNG是反的

    gc.collect()  # 回收全部代垃圾，避免内存泄露

    ####################
    # 测试集存成图片
    ####################

    print('\n\n正在把测试集和标签写入 png 中 ...')

    anchor1_pos_test_onehot = onehot_func(anchor1_pos_test, ATGC)
    anchor1_neg2_test_onehot = onehot_func(anchor1_neg2_test, ATGC)
    anchor2_pos_test_onehot = onehot_func(anchor2_pos_test, ATGC)
    anchor2_neg2_test_onehot = onehot_func(anchor2_neg2_test, ATGC)

    # print('单块大小', anchor1_pos_test_onehot.shape)

    # 合并在一起
    test_pos = np.hstack((anchor1_pos_test_onehot, anchor2_pos_test_onehot)) # 积极&积极 横着合并 dstack 竖着合并 hstack
    test_neg = np.hstack((anchor1_neg2_test_onehot, anchor2_neg2_test_onehot)) # 消极&消极 横着合并 dstack 竖着合并 hstack

    print('合块大小', test_pos.shape)
    print('正在生成PNG ...')

    # 写入一张张图片
    for j in tqdm.trange( len(test_pos), ascii=True ):
        imageio.imwrite('test/pos/'+str(j)+'.png', img_as_ubyte(np.transpose(test_pos[j]))) # 必须转置，因为PNG是反的
        imageio.imwrite('test/neg/'+str(j)+'.png', img_as_ubyte(np.transpose(test_neg[j]))) # 必须转置，因为PNG是反的





########################################
#
# 主函数
#
########################################

# import h5py # 导入工具包 

if __name__ == '__main__':

    # 使用PNG好处很多：体积小，速度快，直观，可以直接用自带的API（效率高）
    data_process()

    print('\n已完成所有操作！')
    input('\n请按任意键继续. . .') # 避免闪退
