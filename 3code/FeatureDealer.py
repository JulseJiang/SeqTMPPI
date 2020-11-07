# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/18 17:15
@desc:
"""
import os
import numpy as np
from keras.utils import to_categorical

from PairDealer import getPairs
from common import check_path, handleBygroup


class Feature_type:
    V_PSSM='V_PSSM'
    H_PSSM='H_PSSM'
    SEQ_1D = 'SEQ_1D'
    SEQ_1D_OH = 'SEQ_1D_OH' # onehot
    SEQ_2D = 'SEQ_2D'
class BaseFeature:
    def base_compose(self,dirout_feature,fin_pair,dir_feature_db,feature_type='V_PSSM',fout_pair=''):
        check_path(dirout_feature)
        fo  = open(fout_pair,'w') if fout_pair!='' else None
        for pairs in getPairs(fin_pair):
            a = pairs[0]
            b = pairs[1]
            # print(pairs)  # ['O35668', 'P00516']
            fa = os.path.join(dir_feature_db, a + '.npy')
            fb = os.path.join(dir_feature_db, b + '.npy')
            if not (os.access(fa, os.F_OK) and os.access(fb, os.F_OK)):
                print('===============features of pairs not found %s %s================' % (a, b), os.access(fa, os.F_OK),
                      os.access(fb, os.F_OK))
                continue
            pa = np.load(fa)
            pb = np.load(fb)
            if (len(pa)<50 or len(pa)>2000 or max(pa)>20) or (len(pb)<50 or len(pb)>2000 or max(pb)>20):
                print('wrong length or x')
                continue
            if fo!=None:
                fo.write('%s\t%s\n'%(a,b))
                fo.flush()
            # padding
            if feature_type == Feature_type.V_PSSM:pc = self.padding_PSSM(pa,pb,vstack=True)
            elif feature_type == Feature_type.H_PSSM:pc = self.padding_PSSM(pa,pb,vstack=False)
            elif feature_type == Feature_type.SEQ_1D:pc = self.padding_seq1D(pa,pb,vstack=False)
            # elif feature_type == Feature_type.SEQ_1D_OH:pc = self.padding_seq1D(pa,pb,vstack=False)
            elif feature_type == Feature_type.SEQ_2D:pc = self.padding_seq2D(pa,pb)
            else:
                print('incoreect feature_type')
                return
            # 保存padding后的成对特征
            fout = os.path.join(dirout_feature, "%s_%s.npy" % (a, b))
            np.save(fout, pc)
            del pc, pa, pb
        if fo != None:
            fo.close()
    def padding_PSSM(self,pa,pb,vstack=True,shape=(2000,21)):
        pa_pad_col = np.pad(pa, ((0, 0), (0, shape[1]-pa.shape[1])), 'constant', constant_values=(0, 1))
        pb_pad_col = np.pad(pb, ((0, 0), (0, shape[1]-pb.shape[1])), 'constant', constant_values=(0, 1))
        # 前期工作不够严谨，估计有超过两千长度的蛋白
        pa_pad_row = np.pad(pa_pad_col, ((0, shape[0] - pa.shape[0]), (0, 0)), 'constant')
        pb_pad_row = np.pad(pb_pad_col, ((0, shape[0] - pb.shape[0]), (0, 0)), 'constant')
        pc = np.vstack([pa_pad_row, pb_pad_row]) if vstack else np.hstack([pa_pad_row, pb_pad_row])
        return pc
    def padding_seq1D(self,pa,pb,vstack=True,shape=(2000,)):
        # data.shape = (4000,)
        # warring padding number not appear in origin data
        pa_pad_col = np.pad(pa, ((0, shape[0]-pa.shape[0])), 'constant', constant_values=0)
        pb_pad_col = np.pad(pb, ((0, shape[0]-pb.shape[0])), 'constant', constant_values=0)
        pc = np.vstack([pa_pad_col, pb_pad_col]) if vstack else np.hstack([pa_pad_col, pb_pad_col])
        return pc
    def padding_seq2D(self,pa,pb,shape=(2000*2000,21*21)):
        pa_pad_col = np.pad(pa, ((0, 2000 - pa.shape[0])), 'constant', constant_values=0)
        pb_pad_col = np.pad(pb, ((0, 2000 - pb.shape[0])), 'constant', constant_values=0)
        pc = np.zeros((2000,2000))
        lookUpTable = self.constructLookUpTable()
        for idx, x in enumerate(pa_pad_col):
            for idy, y in enumerate(pb_pad_col):
                pc[idx, idy] = lookUpTable.index(x*100+y)
                # tmp_sp_2D[i,idx,idy,0] = lookUpTable.index(x*100+y)[0] # todo
        return pc
    # support
    def constructLookUpTable(self):
        lookUpTable = [0] * 21 * 21
        aminos = [i for i in range(21)]
        cell = 0
        for i in aminos:
            for j in aminos:
                lookUpTable[cell] = i * 100 + j
                cell = cell + 1
        return lookUpTable
def getGroupFeature(group_dir_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D):
    src = 'pairdata'
    des = 'feature'
    func = BaseFeature().base_compose
    handleBygroup(group_dir_pair, src, des, func,dir_feature_db,feature_type=feature_type)
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    # group_dir_pair = '/home/jjhnenu/data/PPI/release/data/group/'

if __name__ == '__main__':
    print()
    cloud = 'jjhnenu'

    # filename = 'positiveV1_fswissprot_Composi_5'
    # basepath = '/home/%s/data/PPI/stage2/processPair2445/pair/%s' % (cloud, filename)
    # # dir_std_pssm = '%s/PSSM/output_std_head' % basepath
    '''
    pssm
    '''
    # dir_std_pssm = '/home/jjhnenu/data/PPI/stage2/processPair2445/pair/positiveV1_fswissprot_Composi_5/PSSM/output_std_head'
    # dir_pair = '/home/%s/data/PPI/release/data/p_fp_fw_2_1_1/'%cloud
    # for eachfile in os.listdir(dir_pair):
    #     print(eachfile)
    #     finpair = os.path.join(dir_pair, eachfile)
    #     dir_foutFeaturePair = '/home/%s/data/PPI/release/feature/pssm_feature_2D_vstack/p_fp_fw_2_1_1/%s/' % (cloud,eachfile.split('.')[0])
    #     BaseFeature().compose_PSSM(dir_foutFeaturePair,finpair,dir_std_pssm)
    '''
    seq1D
    '''
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    # dir_pair = '/home/%s/data/PPI/release/data/p_fp_fw_2_1_1/'%cloud
    # for eachfile in os.listdir(dir_pair):
    #     print(eachfile)
    #     fin_pair = os.path.join(dir_pair, eachfile)
    #     dirout_feature = '/home/%s/data/PPI/release/feature/seq_feature_1D/p_fp_fw_2_1_1/%s/' % (cloud,eachfile.split('.')[0])
    #     BaseFeature().base_compose(dirout_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D)
    '''
    seq1D_onehot
    just use seq1D
    '''

    '''
    seq2D
    too large
    '''
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    # dir_pair = '/home/%s/data/PPI/release/data/p_fp_fw_2_1_1/'%cloud
    # for eachfile in os.listdir(dir_pair):
    #     print(eachfile)
    #     fin_pair = os.path.join(dir_pair, eachfile)
    #     dirout_feature = '/home/%s/data/PPI/release/feature/seq_feature_2D/p_fp_fw_2_1_1/%s/' % (cloud,eachfile.split('.')[0])
    #     BaseFeature().base_compose(dirout_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_2D)

    '''
    pssm hstack
    '''
    # dir_feature_db = '/home/jjhnenu/data/PPI/stage2/processPair2445/pair/positiveV1_fswissprot_Composi_5/PSSM/output_std_head'
    # dir_pair = '/home/%s/data/PPI/release/data/p_fp_fw_2_1_1/'%cloud
    # for eachfile in os.listdir(dir_pair):
    #     print(eachfile)
    #     fin_pair = os.path.join(dir_pair, eachfile)
    #     dirout_feature = '/home/%s/data/PPI/release/feature/pssm_feature_2D_hstack/p_fp_fw_2_1_1/%s/' % (cloud,eachfile.split('.')[0])
    #     BaseFeature().base_compose(dirout_feature,fin_pair,dir_feature_db,feature_type=Feature_type.H_PSSM)

    '''
    pssm 400 hstack
    '''
    # dir_feature_db = '/home/jjhnenu/data/PPI/stage2/processPair2445/pair/positiveV1_fswissprot_Composi_5/PSSM/output_std_400_1'
    # dir_pair = '/home/%s/data/PPI/release/data/p_fp_fw_2_1_1/'%cloud
    # for eachfile in os.listdir(dir_pair):
    #     print(eachfile)
    #     fin_pair = os.path.join(dir_pair, eachfile)
    #     dirout_feature = '/home/%s/data/PPI/release/feature/pssm400_feature_1D/p_fp_fw_2_1_1/%s/' % (cloud,eachfile.split('.')[0])
    #     BaseFeature().base_compose(dirout_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D)

    '''
    group seq1D
    '''
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    # group_dir_pair = '/home/jjhnenu/data/PPI/release/pairdata/group/'
    # getGroupFeature(group_dir_pair, dir_feature_db, feature_type=Feature_type.SEQ_1D)


    '''
    feature pair 
    positive 
    swissprot
    '''
    dir_feature_db = '/home/19jiangjh/data/PPI/release/featuredb/seq_feature_1D/'
    dir_pair = '/home/19jiangjh/data/PPI/release/pairdata'
    for eachfile in ['negative_fswissprot_7177.txt','negative_fpositive_10245.txt','positive_2049.txt']:
        print(eachfile)
        fin_pair = os.path.join(dir_pair, eachfile)
        dirout_feature = '/home/19jiangjh/data/PPI/release/feature/p_fp_fw'
        BaseFeature().base_compose(dirout_feature,fin_pair,dir_feature_db,feature_type=Feature_type.SEQ_1D)