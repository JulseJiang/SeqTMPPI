# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/15 21:33
@desc:
"""
import os

# from myData import TmpPSSMData
# from myModel import MyModel, Param
import time

from keras import models

from common import check_path, handleBygroup, pair_ID, Feature_pair_DB
from myData import TmpPSSMData, BaseData
from myEvaluate import MyEvaluate
from myModel import MyModel, Param
from mySupport import calculateResults, savepredict, plot_result
import pandas as pd

def getGroupResult(group_dir_pair,model_type = Param.CNN1D,limit=0,onehot=False,des = 'result_embedding'):
    src = 'pairdata'
    # des = 'result_embedding'
    func = entry
    dir_in = group_dir_pair.replace(src, 'feature')
    handleBygroup(group_dir_pair, src, des, func,dir_in,model_type = model_type,limit=limit,onehot=onehot)
def entry(dirout,fin_pair,dir_in,model_type = Param.CNN1D,limit=0,onehot=False,kernel_size = 3,epochs=60,filters = 250,batch_size = 100,validate=None):
    # model_type = Param.CNN1D
    # fin_pair =  '/home/jjhnenu/data/PPI/release/data/p_fp_fw_2_1_1/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/pssm400_feature_1D/p_fp_fw_2_1_1/all/'
    # fout = '/home/jjhnenu/data/PPI/release/result/pssm400_feature_1D/p_fp_fw_2_1_1/all/'
    # if 'all' not in dirout:return
    check_path(dirout)
    print('dirout:',dirout)
    # dir_in = dirout.replace(des, 'feature')
    print('dir_in:',dir_in)
    print('fin_pair',fin_pair)
    bd = BaseData()
    if validate == None:
        (x_train, y_train), (x_test, y_test) = bd.load(fin_pair,dir_in,test_size=0.1,limit=limit,onehot=onehot)
    else:
        x_test, y_test = bd.loadTest(validate['fin_pair'],validate['dir_in'],onehot=onehot,is_shuffle=True)
        x_train, y_train= bd.loadTest(fin_pair,dir_in,onehot=onehot,is_shuffle=True)

    print('Build and fit model...')
    print('x_train.shape[1:]',x_train.shape[1:])
    mm = MyModel(model_type = model_type,
                    input_shape = x_train.shape[1:],
                    filters = filters,
                    kernel_size = kernel_size,
                    pool_size = 2,
                    hidden_dims = 250,
                    batch_size = batch_size,
                    epochs = epochs,)
    mm.process(dirout,x_train, y_train, x_test, y_test)
    print('save result to %s'%dirout)
def main(limit=5):
# if __name__ == '__main__':
#     limit = 0
    # limit = 5
    onehot = False
    '''
    load feature
    '''
    print('Loading feature...')
    '''
    pssm
    '''
    # model_type = Param.CNN2D
    # tpssmd = TmpPSSMData()
    # # (x_train, y_train), (x_test, y_test) = tpssmd.loadPAN(inPDir, inNDir, limit=limit)
    #
    # fin_pair =  '/home/jjhnenu/data/PPI/release/data/p_fp_fw_2_1_1/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/pssm_feature_2D/p_fp_fw_2_1_1/all/'
    # fout = '/home/jjhnenu/data/PPI/release/result/pssm_feature_2D/p_fp_fw_2_1_1/all'
    # check_path(fout)
    # (x_train, y_train), (x_test, y_test) = tpssmd.load(fin_pair,dir_in,limit=limit)
    '''
    seq1D
    '''
    # model_type = Param.CNN1D
    # fin_pair =  '/home/jjhnenu/data/PPI/release/data/p_fp_fw_2_1_1/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/all/'
    # fout = '/home/jjhnenu/data/PPI/release/result/seq_feature_1D/p_fp_fw_2_1_1/all/'
    '''
    seq1D onehot
    '''
    # model_type = Param.CNN1D_OH
    # onehot = True
    # fin_pair =  '/home/jjhnenu/data/PPI/release/data/p_fp_fw_2_1_1/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/all'
    # fout = '/home/jjhnenu/data/PPI/release/result/seq_feature_1D_onehot/p_fp_fw_2_1_1/all/'
    '''
    test seq2D
    Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
    2020-04-30 21:41:50.315008: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
    2020-04-30 21:41:50.315029: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly
    '''
    onehot= True
    model_type = Param.CNN2D
    fin_pair =  '/home/jjhnenu/data/PPI/release/pairdata/p_fp_fw_2_1_1/all.txt'
    dir_in =  '/home/jjhnenu/data/PPI/release/feature/seq_feature_2D/p_fp_fw_2_1_1/all/'
    fout = '/home/jjhnenu/data/PPI/release/result/seq_feature_2D/p_fp_fw_2_1_1/all/'
    '''
    pssm hstack
    '''
    # model_type = Param.CNN2D
    # fin_pair =  '/home/jjhnenu/data/PPI/release/data/p_fp_fw_2_1_1/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/pssm_feature_2D_hstack/p_fp_fw_2_1_1/all/'
    # fout = '/home/jjhnenu/data/PPI/release/result/pssm_feature_2D_hstack/p_fp_fw_2_1_1/all/'
    '''
    pssm 400
    '''
    # model_type = Param.CNN1D
    # fin_pair =  '/home/jjhnenu/data/PPI/release/data/p_fp_fw_2_1_1/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/pssm400_feature_1D/p_fp_fw_2_1_1/all/'
    # fout = '/home/jjhnenu/data/PPI/release/result/pssm400_feature_1D/p_fp_fw_2_1_1/all/'

    check_path(fout)
    (x_train, y_train), (x_test, y_test) = BaseData().load(fin_pair,dir_in,limit=limit,onehot=onehot)
    print('Build and fit model...')
    print('x_train.shape[1:]',x_train.shape[1:])
    mm = MyModel(model_type = model_type,
                    input_shape = x_train.shape[1:],
                    filters = 250,
                    kernel_size = 3,
                    pool_size = 2,
                    hidden_dims = 250,
                    batch_size = 100,
                    epochs = 80,)
    mm.process(fout,x_train, y_train, x_test, y_test)
    print('save result to %s'%fout)

if __name__ == '__main__':
    print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()

    # cloud = 'jjhnenu'
    # cloud = '19jiangjh'
    # fout = '/home/%s/data/PPI/stage2/processPair2445/pair/positiveV1/PSSM/result_4000_21/0/'%cloud
    # inPDir = '/home/%s/data/PPI/stage2/processPair2445/pair/positiveV1/PSSM/feature/positive_4000_21'%cloud
    # inNDir = '/home/%s/data/PPI/stage2/processPair2445/pair/positiveV1/PSSM/feature/negative_4000_21/0'%cloud
    #
    # model_type = Param.CNN2D

    # fin_pair =  '/home/jjhnenu/data/PPI/release/feature/pssm_feature_2D/all.txt'
    # dir_in =  '/home/jjhnenu/data/PPI/release/feature/pssm_feature_2D/'
    # fout = '/home/jjhnenu/data/PPI/release/result/p_fp_fw_2_1_1/all/'
    # model_type = Param.CNN2D
    # model_type = Param.CNN1D
    # main(model_type=model_type)

    # main()


    # group_dir_pair = '/home/jjhnenu/data/PPI/release/pairdata/group/'
    # model_type = Param.CNN1D
    # onehot = False
    # getGroupResult(group_dir_pair, model_type=model_type, limit=0, onehot=onehot)

    '''
    test kernel size
    '''
    # kernel_size = [x * 1 for x in range(25, 35)]
    # kernel_size = [x * 6 for x in range(2, 9)]
    # for k in kernel_size:
    #     dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_single/all/',str(k))
    #     fin_pair = '/home/jjhnenu/data/PPI/release/pairdata/p_fp_fw_2_1_1/all.txt'
    #     dir_in = '/home/jjhnenu/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/all/'
    #     onehot = True
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=k)

    # k = 3
    # dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_kernalsize/all/',str(k))
    # fin_pair = '/home/jjhnenu/data/PPI/release/pairdata/p_fp_fw_2_1_1/all.txt'
    # dir_in = '/home/jjhnenu/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/all/'
    # onehot = True
    # entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=k)

    # dirin = '/home/jjhnenu/data/PPI/release/result_kernalsize/all/'
    # dirout = '/home/jjhnenu/data/PPI/release/statistic/result_kernalsize/all/'

    # dirin = '/home/jjhnenu/data/PPI/release/result_single/all'
    # dirout = dirin
    # calculateResults(dirout,dirin)

    '''
    test epoch
    '''
    # epoch = [x * 10 +60 for x in range(1,9)]
    # for e in epoch:
    #     dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_epoch/all/',str(e))
    #     fin_pair = '/home/jjhnenu/data/PPI/release/pairdata/p_fp_fw_2_1_1/all.txt'
    #     dir_in = '/home/jjhnenu/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/all/'
    #     onehot = True
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, epochs = e)

    '''
    test cnn+lstm 
    '''
    # dirout = '/home/jjhnenu/data/PPI/release/result/seq_feature_1D_CNN_LSTM/p_fp_fw_2_1_1/all/'
    # fin_pair = '/home/jjhnenu/data/PPI/release/pairdata/p_fp_fw_2_1_1/all.txt'
    # dir_in = '/home/jjhnenu/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/all/'
    # onehot = False
    # entry(dirout, fin_pair, dir_in, model_type=Param.CNN_LSTM, limit=0, onehot=onehot,kernel_size = 10,epochs=20,filters = 64,batch_size = 128)

    '''
    used in paper
    dataset:		p_fp 
    form:			onehot
                    'samples': 3688
                    "batch_input_shape": [null, 4000, 21],

    Param:			
                    "filters": 250, 
                    "kernel_size": [21]
                    'batch_size': 150, 
                    'epochs': 80
    
    group result:	*
    '''
    # base_dirin = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_in_paper/standard_experiment/',eachfile)
    #     dir_in = os.path.join('/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=21, epochs=80,
    #           filters=250, batch_size=150)
    '''
    alter param
    best kernel size = 90
    '''
    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     onehot = True
    #     base_dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_kernel_size/'
    #     kernel_size = [x * 9 for x in range(1, 11)]
    #     for k in kernel_size:
    #         dirout = os.path.join(base_dirout,str(k),
    #                               eachfile)
    #         check_path(dirout)
    #         entry(dirout, fin_pair, dir_in,
    #               model_type=Param.CNN1D_OH,
    #               limit=0,
    #               onehot=onehot,
    #               kernel_size=k,
    #               epochs=80,
    #               filters=250,
    #               batch_size=150)

    '''
    test single
    '''
    # # eachfile = str(0)
    # kernel_size = 99
    # filters = 250
    # batch_size = 150
    # base_dirin = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join('/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/', eachfile, 'all.txt')
    #     dir_in = os.path.join('/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/', eachfile, 'all')
    #     base_dirout = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_k_99_f250_b150'
    #     onehot = True
    #     times = [x for x in range(0, 10)]
    #     for k in times:
    #         dirout = os.path.join(base_dirout, str(k),
    #                             eachfile)
    #         check_path(dirout)
    #         entry(dirout, fin_pair, dir_in,
    #             model_type=Param.CNN1D_OH,
    #             limit=0,
    #             onehot=onehot,
    #             kernel_size=kernel_size,
    #             epochs=80,
    #             filters=filters,
    #             batch_size=batch_size)
    '''
    kernel size = 90
    best filters num = 
    '''
    '''
    test single
    '''
    # eachfile = str(0)
    # fin_pair = os.path.join('/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/', eachfile, 'all.txt')
    # dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/', eachfile, 'all')
    # base_dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_kernel_size_27/'
    # onehot = True
    # kernel_size = [x * 27 for x in range(4, 11)]
    # for k in kernel_size:
    #     dirout = os.path.join(base_dirout, str(k),
    #                         eachfile)
    #     check_path(dirout)
    #     entry(dirout, fin_pair, dir_in,
    #         model_type=Param.CNN1D_OH,
    #         limit=0,
    #         onehot=onehot,
    #         kernel_size=k,
    #         epochs=80,
    #         filters=250,
    #         batch_size=150)
    '''
    opotimize
    kernelsize 90 0.44
    filters num 200 0.44
    batch size 90 0.39
    epoch 80
    '''
    # base_dirin = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     if eachfile == '0':
    #         print('eachfile',eachfile)
    #         continue
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dir_in = os.path.join('/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/', eachfile, 'all')
    #     dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_kernel_size/early_stop/opotimize_90_200_90_group', eachfile, 'all')
    #     onehot = True
    #     check_path(dirout)
    #     entry(dirout, fin_pair, dir_in,
    #           model_type=Param.CNN1D_OH,
    #           limit=0,
    #           onehot=onehot,
    #           kernel_size=90,
    #           epochs=80,
    #           filters=200,
    #           batch_size=90)

    '''
    batch size
    '''
    # base_dirin = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dir_in = os.path.join('/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     onehot = True
    #     base_dirout = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_kernel_size/'
    #     kernel_size = [x * 30 for x in range(2, 8)]
    #     for k in kernel_size:
    #         dirout = os.path.join(base_dirout,str(k),
    #                               eachfile)
    #         check_path(dirout)
    #         entry(dirout, fin_pair, dir_in,
    #               model_type=Param.CNN1D_OH,
    #               limit=0,
    #               onehot=onehot,
    #               kernel_size=21,
    #               epochs=80,
    #               filters=250,
    #               batch_size=k)

    '''
    long distance kernel on GPU
    '''
    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     onehot = True
    #     base_dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_kernel_size_32/'
    #     kernel_size = [x * 32 for x in range(1, 5)]
    #     for k in kernel_size:
    #         dirout = os.path.join(base_dirout,str(k),
    #                               eachfile)
    #         check_path(dirout)
    #         entry(dirout, fin_pair, dir_in,
    #               model_type=Param.CNN1D_OH,
    #               limit=0,
    #               onehot=onehot,
    #               kernel_size=k,
    #               epochs=80,
    #               filters=250,
    #               batch_size=150)
    '''
    test filter 
    '''
    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     onehot = True
    #     base_dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_filters/'
    #     kernel_size = [x * 50 for x in range(2, 6)]
    #     for k in kernel_size:
    #         dirout = os.path.join(base_dirout,str(k),
    #                               eachfile)
    #         check_path(dirout)
    #         entry(dirout, fin_pair, dir_in,
    #               model_type=Param.CNN1D_OH,
    #               limit=0,
    #               onehot=onehot,
    #               kernel_size=k,
    #               epochs=80,
    #               filters=250,
    #               batch_size=150)
    '''
    test maxpooling
    '''
    # base_dirin = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_in_paper/alter_network/alter_maxpooling/',eachfile)
    #     dir_in = os.path.join('/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_MAX_OH, limit=0, onehot=onehot, kernel_size=21, epochs=80,
    #           filters=250, batch_size=150)

    '''
    test GlobalMaxPooling1D on GPU
    '''
    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dirout = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_network/alter_GlobalMaxPooling1D/',eachfile)
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_MAX_OH, limit=0, onehot=onehot, kernel_size=21, epochs=80,
    #           filters=250, batch_size=150)
    '''
    test maxpooling and GlobalMaxPooling1D on GPU
    '''
    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/'
    # dirout = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_network/alter_Global_CNN_MaxPooling1D/',str(0))
    # dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/',str(0),'all')
    # fin_pair = os.path.join(base_dirin,str(0),'all.txt')
    # onehot = True
    # entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_MAX_OH, limit=0, onehot=onehot, kernel_size=21, epochs=80,
    #           filters=250, batch_size=150)
    '''
    test maxpooling and gloabal average pooling1D
    '''
    # base_dirin = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/'
    # dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_in_paper/alter_network/alter_Max_GlobalAveragePooling1D/',str(0))
    # dir_in = os.path.join('/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/',str(0),'all')
    # fin_pair = os.path.join(base_dirin,str(0),'all.txt')
    # onehot = True
    # entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_MAX_OH, limit=0, onehot=onehot, kernel_size=21, epochs=80,
    #           filters=250, batch_size=150)

    # base_dirin = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dirout = os.path.join(
    #         '/home/jjhnenu/data/PPI/release/result_in_paper/alter_network/alter_Max_GlobalAveragePooling1D/', str(0))
    #     dir_in = os.path.join('/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/', str(0), 'all')
    #     onehot = True
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_MAX_OH, limit=0, onehot=onehot, kernel_size=21,
    #           epochs=80,
    #           filters=250, batch_size=150)


    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dirout = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_network/alter_Max_GlobalAveragePooling1D/',eachfile)
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/',eachfile,'all')
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_MAX_OH, limit=0, onehot=onehot, kernel_size=21, epochs=80,
    #           filters=250, batch_size=150)

    '''
    change dataset
    '''
    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_fw_2_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin,eachfile,'all.txt')
    #     dirout = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_dataset/compare_standard/p_fp_fw_2_1_1/',eachfile)
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_fw_2_1_1/',eachfile,'all')
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=21, epochs=80,
    #           filters=250, batch_size=150)
    #
    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fw_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin, eachfile, 'all.txt')
    #     dirout = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_dataset/compare_standard/p_fw_1_1/',
    #                           eachfile)
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fw_1_1/', eachfile, 'all')
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=21,
    #           epochs=80,
    #           filters=250, batch_size=150)

    # base_dirin = '/home/19jiangjh/data/PPI/release/pairdata/group/p_fp_1_1/'
    # for eachfile in os.listdir(base_dirin):
    #     fin_pair = os.path.join(base_dirin, eachfile, 'all.txt')
    #     dirout = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_dataset/p_fp_1_1/',
    #                           eachfile)
    #     dir_in = os.path.join('/home/19jiangjh/data/PPI/release/feature/group/p_fp_1_1/', eachfile, 'all')
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=21,
    #           epochs=80,
    #           filters=250, batch_size=150)

    # k = (2,3,5,7)
    # dirout = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_kernel_size','kernel_size_2357')
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fp_1_1/all.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/seq_feature_1D/p_fp_1_1/all/'
    # onehot = True
    # entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=k)
    '''
    can't run
    need to add several model with different kernel size,and ensomble them.
    '''
    # k = (2, 3, 5, 7)
    # dirout = os.path.join('/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_kernel_size',
    #                       'kernel_size_2357')
    # fin_pair = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/0/all.txt'
    # dir_in = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/0/all/'
    # onehot = True
    # entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=3, onehot=onehot, kernel_size=k)
    '''
    test on model
    '''
    # # fin_pair = '/home/jjhnenu/data/PPI/release/otherdata/positive_2049/TMP_SP/TMP_SP_1395.txt'
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/1/0/test.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # # fin_pair = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/TMP_SP/TMP_nonTMP_1156.txt'
    # # dir_in = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/0/all'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/_my_model.h5'
    # onehot = True
    # (x_train, y_train), (x_test, y_test) = BaseData().load(fin_pair, dir_in, limit=0, onehot=onehot)
    # model = models.load_model(fin_model,custom_objects=MyEvaluate.metric_json)
    # result = model.evaluate(x_test, y_test, verbose=False)
    # # result = model.predict(x_test)
    # print('Loss:%f,ACC:%f'%(result[0],result[1]))
    # # [0.5745526841708593, 0.550000011920929, 1.0, 0.5333333611488342, 0.6880943179130554, 0.0]
    # # [0.48189975154109116, 0.74634146225161668, 0.87272867807527865, 0.5661648439198006, 0.68134220227962583, 0.52066026286381051]
    # # [0.47267651368932029, 0.76829268176381182, 0.92864025220638369, 0.60068764541207287, 0.71928398318407016, 0.58933141522291232]


    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/1/0/all.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_17422'
    # dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio'

    # for idx in range(1,11):
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/%d/0/train.txt'%idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/%d'%idx
    #     check_path(dirout)
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=99,
    #           epochs=80,
    #           filters=300, batch_size=90)

    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_1'
    # dirout = dirin
    # calculateResults(dirout,dirin)



    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/1/0/test.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/test'


    # fin_pair = '/home/jjhnenu/data/PPI/release/pairdata/p_fw/1/0/test.txt'
    # dir_in = '/home/jjhnenu/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw/1/_my_model.h5'
    # dirout_result = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw/1/test'

    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/1/0/test.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/test'
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)
    '''
    different ratio
    '''
    # for idx in range(1,11):
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/0/train.txt'%idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_1/%d'%idx
    #     check_path(dirout)
    #     validate = {}
    #     validate['fin_pair']  = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/0/validate.txt'%idx
    #     validate['dir_in'] = dir_in
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=99,
    #           epochs=80,
    #           filters=300, batch_size=90,validate=validate)

    '''
    different set on ratio
    '''
    # for idx in range(1,11):
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/0/train.txt' % idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/_my_model.h5'%idx
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/train'%idx
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/0/validate.txt' % idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/_my_model.h5' % idx
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/validate' % idx
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/0/test.txt' % idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/_my_model.h5' % idx
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/test' % idx
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)

    # idx = 1
    # /home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw/10
    # p_fw_v1_train_validate_v2_fixpositive



    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/0/validate.txt' % idx
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/_my_model.h5' % idx
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/%d/validate_v1' % idx
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)


    # for idx in range(1,11):
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/%d/0/train0.txt'%idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     dirout = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/%d'%idx
    #     check_path(dirout)
    #     validate = {}
    #     validate['fin_pair']  = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/%d/0/validate0.txt'%idx
    #     validate['dir_in'] = dir_in
    #     onehot = True
    #     print(dirout)
    #     entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=99,
    #           epochs=80,
    #           filters=300, batch_size=90,validate=validate)
    #
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/%d/0/train0.txt' % idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/%d/_my_model.h5' % idx
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/%d/train' % idx
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/%d/0/validate0.txt' % idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/%d/_my_model.h5' % idx
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/%d/validate' % idx
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/%d/0/test0.txt' % idx
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/%d/_my_model.h5' % idx
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/%d/test' % idx
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixpositive/'
    # dirout = dirin
    # calculateResults(dirout, dirin)

    '''
    test False positive,wait for test
    '''
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fp_1_1/all.txt'
    #     # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2/_my_model.h5'
    #     # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2/test_p_fp_all/'
    #     # savepredict(fin_pair, dir_in, fin_model, dirout_result)

    '''
    result on test datset 
    '''
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_fixtotal/'
    # # dirin = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/'
    # dirout = dirin
    # calculateResults(dirout, dirin,filename='/test/log.txt',row = 2,resultfilename='result_on_test.csv')
    '''
    test model on DIP
    '''
    # pair_dir = '/home/19jiangjh/data/PPI/release/otherdata/DIP/_TMP_nonTMP_qualified'
    # dir_in = '/home/19jiangjh/data/PPI/release/otherdata/DIP/_feature/'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2/_my_model.h5'
    # for eachfile in os.listdir(pair_dir):
    #     subdir = eachfile.split('.')[0]
    #     fin_pair = os.path.join(pair_dir,eachfile)
    #     dirout_result = os.path.join('/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2/test_DIP',subdir)
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)

    '''
    test on p_fp
    '''
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fp_1_1/all.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/test_p_fp_all/'
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)
    '''
    test on p_fp_fw
    '''
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fp_fw_2_1_1/all.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/test_p_fp_fw_all/'
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #
    '''
    test on p_fw
    '''
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_1_1/all.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/test_p_fw_all/'
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)

    '''
    test on other set
    '''
    # dirin_pair = '/home/19jiangjh/data/PPI/release/otherdata/DIP/_TMP_nonTMP_qualified_drop_positive'
    # dir_in = '/home/19jiangjh/data/PPI/release/otherdata/DIP/_feature/'
    # for eachfile in os.listdir(dirin_pair):
    #     fin_pair = os.path.join(dirin_pair,eachfile)
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_p_fw/9/2/_my_model.h5'
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_p_fw/9/2/dropPositive_test_%s/'%eachfile.split('.')[0]
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)

    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_p_fw/9/2'
    # dirout = dirin
    # calculateResults(dirout,dirin,filename='log.txt',row = 2,resultfilename = 'result.csv')

    '''
    test on Imex 20200708
    '''
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_1_1/all.txt'
    # # fin_pair = '/home/19jiangjh/data/PPI/release/otherdata/uniprotPiarFromImex20200709/TMP_nonTMP_drop_positive_qualified_958.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/1/test_IMEx20200708/'
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)
    '''
    test on GPU
    '''
    # fin_pair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPiarFromImex20200709/TMP_nonTMP_drop_positive_qualified_958.txt'
    # dir_in = '/home/jjhnenu/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2/_my_model.h5'
    # dirout_result = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2/test_IMEx20200708/'
    # '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2'
    # savepredict(fin_pair, dir_in, fin_model, dirout_result)

    # dirin = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_p_fw/9/2'
    # dirout = dirin
    # calculateResults(dirout,dirin,filename='log.txt',row = 2,resultfilename = 'result.csv')

    '''
    release 2 
    standard
    benchmark dataset p_fw_1_1 5 group
    '''
    # for idx in range(0,5):
    #     try:
    #         fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/train.txt'%idx
    #         dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #         dirout = '/home/19jiangjh/data/PPI/release/result_in_paper_2/standard/%d'%idx
    #         check_path(dirout)
    #         validate = {}
    #         validate['fin_pair']  = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/validate.txt'%idx
    #         validate['dir_in'] = dir_in
    #         onehot = True
    #         print(dirout)
    #         entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=21,
    #               epochs=80,
    #               filters=250, batch_size=150,validate=validate)
    #     except:
    #         print('plot wrong!')

    '''
    plot result 
    '''
    # fin = '/home/19jiangjh/data/PPI/release/result_in_paper_2/standard/0/_history_dict.txt'
    # outdir = '/home/19jiangjh/data/PPI/release/result_in_paper_2/standard/0/'
    # with open(fin) as fi:
    #     line = fi.readline()
    #     mydict = eval(line)
    #     plot_result(mydict, outdir)
    '''
    alter param
    alter kernel size
    9 for 9 -117 
    9*1-13
    9487.748238563538 2-14
    '''
    for k in range(1,14):
        for idx in range(0,5):
            if idx==0:continue
            fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/train.txt'%idx
            dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
            dirout = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_param/kernelsize/%d/%d'%(k*9,idx)
            check_path(dirout)
            validate = {}
            validate['fin_pair']  = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/validate.txt'%idx
            validate['dir_in'] = dir_in
            onehot = True
            print(dirout)
            entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=k*9,
                  epochs=80,
                  filters=250, batch_size=150,validate=validate)
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_param/kernelsize'
    # dirout = dirin
    # calculateResults(dirout, dirin, filename='0/_evaluate.txt', row=0, resultfilename='result.csv')
    '''
    alter param
    alter filters
    50 for 1-7
    5389.574913024902
    300 is the best 
    '''
    # for f in range(1,8):
    #     for idx in range(0,5):
    #         fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/train.txt'%idx
    #         dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #         dirout = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_param/filters/%d/%d'%(f*50,idx)
    #         check_path(dirout)
    #         validate = {}
    #         validate['fin_pair']  = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/validate.txt'%idx
    #         validate['dir_in'] = dir_in
    #         onehot = True
    #         print(dirout)
    #         entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
    #               epochs=80,
    #               filters=f*50, batch_size=150,validate=validate)
    #         break
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_param/filters'
    # dirout = dirin
    # calculateResults(dirout, dirin, filename='0/_evaluate.txt', row=0, resultfilename='result.csv')
    '''
   alter param
   alter batch size
   10 6-17
   9*1-13
   '''
    # for b in range(6, 17):
    #     for idx in range(0, 5):
    #         fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/train.txt' % idx
    #         dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #         dirout = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_param/batch_size/%d/%d' % (b*10, idx)
    #         check_path(dirout)
    #         validate = {}
    #         validate['fin_pair'] = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/1/%d/validate.txt' % idx
    #         validate['dir_in'] = dir_in
    #         onehot = True
    #         print(dirout)
    #         entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
    #               epochs=80,
    #               filters=300, batch_size=b*10, validate=validate)
    #         break
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_param/batch_size'
    # dirout = dirin
    # calculateResults(dirout, dirin, filename='0/_evaluate.txt', row=0, resultfilename='result.csv')
    '''
    training with different ratio
    22138.141463279724
    '''
    # for ratio in range(1, 11):
    #     for idx in range(0, 5):
    #         fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/%d/train.txt' % (ratio,idx)
    #         dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #         dirout = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_ratio/%d/%d' % (ratio, idx)
    #         check_path(dirout)
    #         validate = {}
    #         validate['fin_pair'] = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/%d/validate.txt' % (ratio,idx)
    #         validate['dir_in'] = dir_in
    #         onehot = True
    #         print(dirout)
    #         entry(dirout, fin_pair, dir_in, model_type=Param.CNN1D_OH, limit=0, onehot=onehot, kernel_size=90,
    #               epochs=80,
    #               filters=300, batch_size=70, validate=validate)
    #         break
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_ratio'
    # dirout = dirin
    # calculateResults(dirout, dirin, filename='0/_evaluate.txt', row=0, resultfilename='result.csv')
    '''
    predict on test
    '''
    # ratio=3
    # for idx in range(0,5):
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/%d/test.txt' % (ratio,idx)
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_ratio/%d/%d/_my_model.h5'% (ratio,idx)
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper_2/test_on_final/%d/%d/test'% (ratio,idx)
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #     break
    # '''train'''
    # ratio=3
    # for idx in range(0,5):
    #     fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw_v1/%d/%d/train.txt' % (ratio,idx)
    #     dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_ratio/%d/%d/_my_model.h5'% (ratio,idx)
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper_2/test_on_final/%d/%d/train'% (ratio,idx)
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #     break

    # '''
    # test model on DIP
    # '''
    # pair_dir = '/home/19jiangjh/data/PPI/release/otherdata/DIP/_TMP_nonTMP_qualified_drop_positive'
    # dir_in = '/home/19jiangjh/data/PPI/release/otherdata/DIP/_feature/'
    # for eachfile in os.listdir(pair_dir):
    #     subdir = eachfile.split('.')[0]
    #     fin_pair = os.path.join(pair_dir,eachfile)
    #     ratio = 3
    #     for idx in range(0, 5):
    #         fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_ratio/%d/%d/_my_model.h5' % (
    #         ratio, idx)
    #         dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper_2/test_on_final/%d/%d/%s' % (ratio,idx,subdir)
    #         savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #         break
    # #
    # '''
    # test model on IMEXs newly
    # '''
    # ratio = 3
    # fin_pair = '/home/19jiangjh/data/PPI/release/otherdata/uniprotPiarFromImex20200709/TMP_nonTMP_drop_positive_qualified_958.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # for idx in range(0, 5):
    #     fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper_2/alter_ratio/%d/%d/_my_model.h5' % (
    #         ratio, idx)
    #     dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper_2/test_on_final/%d/%d/IMEx_newly' % (ratio, idx)
    #     savepredict(fin_pair, dir_in, fin_model, dirout_result)
    #     break
    '''
    calculate reault 
    '''
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper_2/test_on_final/3/0'
    # dirout = dirin
    # calculateResults(dirout, dirin, filename='log.txt', row=2, resultfilename='result.csv')


    print('stop', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('time', time.time() - start)