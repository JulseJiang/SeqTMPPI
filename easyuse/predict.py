# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/11/15 20:20
@desc:
"""
from FastaDealer import FastaDealer
from FeatureDealer import BaseFeature, Feature_type
from common import check_path
from mySupport import savepredict

if __name__ == '__main__':
    ############# predict #######################################
    ## set file path: protein pair without label
    # in_pair = 'file/source/test1.txt'
    # in_fasta = 'file/source/test1.fasta'

    ############# evaluate #######################################
    # set file path: protein pair with label
    in_pair = 'file/source/test2.txt'
    in_fasta = 'file/source/test2.fasta'

    #######################################################

    fin_model = 'file/source/_my_model.h5'
    out_feature_dir = 'file/feature'
    out_feature_db_dir = 'file/featuredb'
    out_result_dir = 'file/result'

    # check path
    check_path(out_feature_dir)
    check_path(out_result_dir)
    # encoding seq
    FastaDealer().getNpy(in_fasta,out_feature_db_dir)
    # generate feature
    BaseFeature().base_compose(out_feature_dir, in_pair, out_feature_db_dir, feature_type=Feature_type.SEQ_1D)
    # predict and save
    savepredict(in_pair, out_feature_dir, fin_model, out_result_dir)
