# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/7/2 11:24
@desc:
"""
import os

from keras import models

from FastaDealer import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from common import check_path, handledir, countpair
from myData import BaseData
from myEvaluate import MyEvaluate




def process(fin_fasta,dir_feature_db,dir_feature,dir_pair):
    '''
    generate featuredb and feature(feature pair)
    :param fin_fasta:>ID\nseq\n
    :param dir_feature_db:
    :param dir_feature:
    :param dir_pair: contains  several protein file
            # ID pair proteinA\tproteinB\n
    :return:
    '''
    check_path(dir_feature_db)
    check_path(dir_feature)
    # fasta to feature
    fd = FastaDealer()
    fd.getNpy(fin_fasta, dir_feature_db)

    for eachfile in os.listdir(dir_pair):
        print(eachfile)
        fin_pair = os.path.join(dir_pair, eachfile)
        BaseFeature().base_compose(dir_feature, fin_pair, dir_feature_db, feature_type=Feature_type.SEQ_1D)

def process_model(fin_pair,fin_model,dir_feature):
    # fin_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP/Ecoli_TMP_nonTMP_1156.txt'
    # fin_model = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_cpu/9/1/_my_model.h5'
    onehot = True
    (x_train, y_train), (x_test, y_test) = BaseData().load(fin_pair, dir_feature, limit=0, onehot=onehot)
    model = models.load_model(fin_model,custom_objects=MyEvaluate.metric_json)
    result = model.evaluate(x_train, y_train, verbose=False)
    print('Loss:%f,ACC:%f'%(result[0],result[1]))
    result_predict = model.predict(x_train)
    return [fin_pair,result,result_predict]

if __name__ == '__main__':
    '''
    fasta to feature
    '''
    # >ID\nseq\n
    # fin_fasta = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_DIP_fasta20170301_simple.seq'
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_featuredb'
    # check_path(dir_feature_db)
    # fd = FastaDealer()
    # fd.getNpy(fin_fasta,dir_feature_db)
    '''
    generate feature pair
    50-2000 no X
    '''
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_featuredb'
    # dir_feature = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_feature/'
    # # ID pair proteinA\tproteinB\n
    # dir_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP/'
    # dirout_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP_qualified/'
    # check_path(dir_feature)
    # check_path(dirout_pair)
    # for eachfile in os.listdir(dir_pair):
    #     print(eachfile)
    #     fin_pair = os.path.join(dir_pair, eachfile)
    #     fout_pair = os.path.join(dirout_pair,eachfile)
    #     BaseFeature().base_compose(dir_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D,fout_pair=fout_pair)
    #     countpair(fout_pair)



    # ##########################################################
    '''
    test single dataset on model
    '''
    # fin_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP/Ecoli_TMP_nonTMP_1156.txt'
    # fin_model = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_cpu/9/1/_my_model.h5'
    # onehot = True
    # (x_train, y_train), (x_test, y_test) = BaseData().load(fin_pair, dir_feature, limit=0, onehot=onehot)
    # model = models.load_model(fin_model,custom_objects=MyEvaluate.metric_json)
    # result = model.evaluate(x_train, y_train, verbose=False)
    # print('Loss:%f,ACC:%f'%(result[0],result[1]))
    # result_predict = model.predict(x_train)
    # print(result_predict)

    '''
    test several dataset on one model
    '''
    # fin_model = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_cpu/9/1/_my_model.h5'
    # dirin_pair = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP/'
    # func = process_model
    # result = handledir(dirin_pair,func,fin_model)


    # fin_pair='/home/jjhnenu/data/PPI/release/pairdata/test/p_fp_test.txt'
    # fin_model = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90_cpu/9/1/_my_model.h5'
    # dir_feature = Feature_pair_DB.IMEx1
    # result = process_model(fin_pair,fin_model,dir_feature)



# for eachfile in os.listdir(dir_pair):
    #     print(eachfile)
    #     fin_pair = os.path.join(dir_pair, eachfile)
    #     dirout_feature = '/home/%s/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/%s/' % (cloud,eachfile.split('.')[0])
    #     BaseFeature().base_compose(dirout_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D)

    '''
    fasta to feature
    '''
    # >ID\nseq\n
    # fin_fasta = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPiarFromImex20200709/TMP_nonTMP_drop_positive.fasta'
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    # fd = FastaDealer()
    # fd.getNpy(fin_fasta,dir_feature_db,multi=True)

    '''
    generate feature pair
    50-2000 no X
    '''
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    # dir_feature = '/home/jjhnenu/data/PPI/release/feature/p_fp_fw_19471/'
    # # ID pair proteinA\tproteinB\n
    # check_path(dir_feature)
    # fin_pair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPiarFromImex20200709/TMP_nonTMP_drop_positive_1412.txt'
    # fout_pair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPiarFromImex20200709/TMP_nonTMP_drop_positive_qualified.txt'
    # BaseFeature().base_compose(dir_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D,fout_pair=fout_pair)


