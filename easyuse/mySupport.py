# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/15 22:07
@desc:
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import models

from calculate_performance import calculate_performance
from common import handleBygroup, check_path
from myData import BaseData
from myEvaluate import MyEvaluate


def plot_result(history_dict,outdir):
    for key in history_dict.keys():
        print('%s,%s'%(key,str(history_dict[key][-1])))
        if 'val_' in key:continue
        epochs = range(1, len(history_dict[key]) + 1)
        plt.clf()  # 清除数字
        fig = plt.figure()
        plt.plot(epochs, history_dict[key], 'bo', label='Training %s'%key)
        plt.plot(epochs, history_dict['val_'+key], 'b', label='Validation val_%s' %key)
        plt.title('Training and validation %s'%key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.yticks(np.arange(0,1,0.1))
        plt.legend()
        # plt.show()
        fig.savefig(os.path.join(outdir ,'%s.png' % key))
def saveDataset(foutdir,x_train, y_train, x_test, y_test,onehot=False):
    if onehot:
        x_train = np.array([[np.argmax(one_hot)for one_hot in x_train[i]] for i in range(x_train.shape[0])])
        x_test = np.array([[np.argmax(one_hot)for one_hot in x_test[i]] for i in range(x_test.shape[0])])
    train_data = np.hstack([x_train,y_train.reshape(len(y_train),1)])
    test_data = np.hstack([x_test,y_test.reshape(len(y_test),1)])
    np.savetxt(os.path.join(foutdir,'_train_data.txt'),train_data,fmt='%d',delimiter=',')
    np.savetxt(os.path.join(foutdir,'_test_data.txt'),test_data,fmt='%d',delimiter=',')

def calculateResults(dirout,dirin,filename='_evaluate.txt',row = 0,resultfilename = 'result.csv'):
    """
    %s\%s\_evaluate.txt
    :param dirin: contains a list of \%s\_evaluate.txt
    :return:
    dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_2/2/test_DIP/'
    dirout = dirin
    calculateResults(dirout,dirin,filename='log.txt',row = 2,resultfilename = 'result.csv')

    """
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\stage2\processPair2445\pair\positiveV1\onehot\result'
    check_path(dirout)
    count = 0
    data = []
    # columns = ['loss', 'acc', 'metric_precision', 'metric_recall', 'metric_F1score', 'matthews_correlation']
    columns = ['Loss', 'Acc', 'Precision', 'Recall', 'F1score', 'MCC']
    indexs = []
    print(columns)
    print(dirin)
    for eachdir in os.listdir(dirin):
        print(eachdir)
        if '.' in eachdir :continue
        fin = os.path.join(dirin, eachdir)
        sep = '\\' if '\\' in filename else '/'
        if sep in filename:
            for f in filename.split(sep):
                fin = os.path.join(fin, f)
        else:
            fin = os.path.join(fin,filename)
        # fin = '%s\%s\_evaluate.txt' % (dirin, eachdir)
        if not os.access(fin, os.F_OK):
            print('not access to:',fin)
            continue
        with open(fin, 'r') as fi:
            real_row=0
            while(real_row!=row):
                fi.readline()
                real_row = real_row + 1
            line = fi.readline()[:-1]
            # sum += np.array(line.split(':')[-1][1:-1].split(','))
            line = line.replace('nan','0')
            print('****************',line,'********************')
            data.append(line.split(':')[-1][1:-1].split(','))
            indexs.append(eachdir)
            count = count + 1
            print(str(line.split(':')[-1][1:-1].split(','))[1:-1])
    mydata = pd.DataFrame(data)
    mydata.replace('nan',0,inplace=True)
    t = mydata.apply(pd.to_numeric)
    t.loc['mean'] = t.apply(lambda x: x.mean())
    indexs.append('mean')
    t.index = indexs
    t.columns = columns
    t.sort_index(inplace=True)
    t.to_csv(os.path.join(dirout, resultfilename), index=True, header=True)
    # t.to_csv(os.path.join(dirout, 'result.csv'), index=True, header=True,float_format = '%.3f')

def groupCalculate(dirin,filetype='all'):
    """
    /home/jjhnenu/data/PPI/release/result/group/p_fp_1_1/1/all/_evaluate.txt
    :param dirin: /home/jjhnenu/data/PPI/release/result/group
    :return:
    """
    for eachdir in os.listdir(dirin):
        subdir = os.path.join(dirin,eachdir) # /home/jjhnenu/data/PPI/release/result/group/p_fp_1_1/
        data = []
        columns = ['Loss', 'Acc', 'Precision', 'Recall', 'F1score', 'MCC']
        # columns = ['loss', 'acc', 'metric_precision', 'metric_recall', 'metric_F1score', 'matthews_correlation']
        print(columns)
        for eachsubdir in os.listdir(subdir): # 0 1 2 3 4 5
            fin = os.path.join(subdir,eachsubdir,filetype,'_evaluate.txt')
            # fin = os.path.join(subdir,eachsubdir,filetype,'_history_dict.txt')
            if not os.access(fin, os.F_OK): continue
            with open(fin, 'r') as fi:
                line = fi.readline()[:-1]
                # sum += np.array(line.split(':')[-1][1:-1].split(','))
                data.append(line.split(':')[-1][1:-1].split(','))
                print(str(line.split(':')[-1][1:-1].split(','))[1:-1])
        mydata = pd.DataFrame(data)
        t = mydata.apply(pd.to_numeric)
        t.columns = columns
        t.loc['mean'] = t.apply(lambda x: x.mean())
        dirout = os.path.join(subdir.replace('result','statistic'))
        check_path(dirout)
        # float_format = '%.3f'
        t.sort_index(inplace=True)
        t.to_csv(os.path.join(dirout, 'result.csv'),index=True, header=True)
        print(dirout)

def savepredict(fin_pair,dir_in,fin_model,dirout_result):
    # fin_pair = '/home/19jiangjh/data/PPI/release/pairdata/p_fw/1/0/test.txt'
    # dir_in = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw_19471'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/_my_model.h5'
    # dirout_result = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_ratio/p_fw_train_validate/1/test'
    check_path(dirout_result)
    onehot = True
    dataarray = BaseData().loadTest(fin_pair, dir_in,onehot=onehot,is_shuffle=False)
    x_test, y_test =dataarray
    model = models.load_model(fin_model, custom_objects=MyEvaluate.metric_json)
    result = model.evaluate(x_test, y_test, verbose=False,batch_size=90)

    result_predict = model.predict(x_test,batch_size=90)
    result_predict = result_predict.reshape(-1)

    result_class = model.predict_classes(x_test,batch_size=90)
    result_class = result_class.reshape(-1)

    y_test = y_test.reshape(-1)

    print('Loss:%f,ACC:%f' % (result[0], result[1]))

    df = pd.read_table(fin_pair,header=None)
    # df.columns = ['tmp', 'nontmp']
    df.rename(columns={0: 'tmp', 1: 'nontmp'}, inplace=True)
    df['real_label'] = list(y_test)
    df['predict_label'] = result_class
    df['predict'] = result_predict
    df.to_csv(os.path.join(dirout_result,'result.csv'),index=False)

    result_manual = MyEvaluate().evaluate_manual(y_test, result_predict)
    print('[acc,metric_precision, metric_recall, metric_F1score, matthews_correlation]')
    print(result_manual)
    print('[acc,precision,sensitivity,f1,mcc,aps,aucResults,specificity]')
    result_manual2 =calculate_performance(len(x_test), y_test, result_class, result_predict)
    print(result_manual2)
    with open(os.path.join(dirout_result,'log.txt'),'w') as fo:
        fo.write('test dataset %s\n'%fin_pair)
        fo.write('Loss:%f,ACC:%f\n' % (result[0], result[1]))
        fo.write('evaluate result:'+str(result)+'\n')
        fo.write('manual result:[acc,metric_precision, metric_recall, metric_F1score, matthews_correlation]\n')
        fo.write('manual result:' + str(result_manual) + '\n')
        fo.write('manual result2:[acc,precision,sensitivity,f1,mcc,aps,aucResults,specificity]\n')
        fo.write('manual result2:'+str(result_manual2)+'\n')
        fo.flush()

if __name__ == '__main__':
    print()
    # dirin = '/home/jjhnenu/data/PPI/release/result_epoch80/group'
    # groupCalculate(dirin, filetype='all')

    # dirin = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_kernel_size/early_stop/opotimize_90_200_90_group'
    # dirout = dirin
    # calculateResults(dirout,dirin)
    '''
    calclulate 5 group 
    '''
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_dataset/p_fp_1_1'
    # dirout = dirin
    # calculateResults(dirout,dirin)

    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_dataset/p_fw_1_1'
    # dirout = dirin
    # calculateResults(dirout,dirin)
    # dirin = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_dataset/p_fp_fw_2_1_1'
    # dirout = dirin
    # calculateResults(dirout,dirin)
    '''
    calculate every group result
    '''
    # basedir = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_filters_k_99_f300'
    # for eachdir in os.listdir(basedir):
    #     dirin = os.path.join(basedir,eachdir)
    #     if os.path.isfile(dirin):continue
    #     dirout = dirin
    #     calculateResults(dirout,dirin)
    '''
    extract mean result of each kernel
    '''
    # df = pd.DataFrame()
    # # basedir = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_filters'
    # for eachdir in os.listdir(basedir):
    #     basedir2 = os.path.join(basedir,eachdir)
    #     if os.path.isfile(basedir2):continue
    #     for eachkernel in os.listdir(basedir2):
    #         dirin = os.path.join(basedir2, eachkernel)
    #         if os.path.isfile(dirin):
    #             print(dirin)
    #             data = pd.read_csv(dirin,index_col=0)
    #             if df.empty:
    #                 df = pd.DataFrame(columns = data.columns)
    #             # df.loc[eachdir] = data.iloc[-1].values
    #             df.loc[int(eachdir)] = data.iloc[-1].values
    # df.sort_index(inplace=True)
    # df.to_csv(os.path.join(basedir,'result.csv'))

    fin = r'E:\githubCode\data\PPI\release\result_in_paper_2\alter_param\batch_size\70\0'
    outdir = r'E:\githubCode\data\PPI\release\result_in_paper_2\alter_param\batch_size\70\0'
    with open(fin) as fi:
        line = fi.readline()
        mydict = eval(line)
        plot_result(mydict, outdir)
