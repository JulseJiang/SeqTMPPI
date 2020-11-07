# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/19 18:03
@desc:
"""
import os
def check_path(in_dir):
    if '.' in in_dir:
        in_dir,_ = os.path.split(in_dir)
    if not os.path.exists(in_dir):os.makedirs(in_dir)
def processPair(line,sep):
    return line[:-1].split(sep)
def processTXTbyLine(fin,func,sep,title=True):
    """
    func = processPair
    genre = processTXTbyLine(finPair, func, '\t', title=False)
    :param fin: ID pair
    :param func: process pair
    :param sep:
    :param title:
    :return: [proteins for proteins in genre]
    """
    with open(fin,'r')as fo:
        if title:fo.readline()
        line = fo.readline()
        while (line):
            yield func(line,sep)
            line = fo.readline()
def getPairs(fin,sep='\t',title=False):
    '''
    :param fin: ID1\tID2\n
    :param sep:
    :param title:
    :return: [ID1,ID2]

    '''
    func = processPair
    return processTXTbyLine(fin, func, sep, title=title)

def handleBygroup(group_dir_pair,src,des,func,*args, **kwargs):
    """

    :param group_dir_pair:
    :param src: 'data'
    :param des: 'feature'
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    # dir_feature_db = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    # group_dir_pair = '/home/jjhnenu/data/PPI/release/data/group/'
    for in_dir in os.walk(group_dir_pair):
        if '.' not in str(in_dir): continue
        dir_pair = in_dir[0]
        if dir_pair.count(src)!=1:assert '%s can only appear once'%src
        out_dir = dir_pair.replace(src,des)
        print('out put file to ',out_dir)
        check_path(out_dir)
        for eachfile in in_dir[2]:
            fin_pair = os.path.join(in_dir[0], eachfile)
            dirout_feature = os.path.join(out_dir,eachfile.split('.')[0])
            print(dirout_feature)
            func(dirout_feature,fin_pair,*args, **kwargs)

def readIDlist(filePath,by = '\n'):
    fo = open(filePath,'r')
    lines = fo.readlines()
    # 'P67999\n', 'P0ADV1\n', 'P0A6F5\n',
    list = []
    for line in lines:
        list.append(line.split(by)[0])
    fo.close()
    # 'P67999', 'P0ADV1', 'P0A6F5'
    return list
def saveList(inlist,outListFile,by = '\n',file_mode='w'):
    """
    将list保存为指定格式的txt文件
    :param inlist: 输入的list
    :param outListFile: 输出list文件的位置
    :param by:分割符号
    :return:
    eg
    idlist = [1,2,3]
    outListF = 'file/result/pfam/pfamJson/queryIdList.txt'
    saveList(idlist, outListF, by='\n')
    """
    with open(outListFile,file_mode) as fo:
        for e in inlist:
            fo.write(str(e)+by)
            fo.flush()

'''
for extraTMP-SP pair
'''
def countpair(fin):
    with open(fin,'r') as fo:
        rowsnum = len(fo.readlines())
    indir,fname = os.path.split(fin)
    prename = fname.split('.')[0]
    os.rename(fin,os.path.join(indir,'%s_%d.txt'%(prename,rowsnum)))
def handledir(dirin,func,*args):
    result_list = []
    for eachfile in os.listdir(dirin):
        result_list.append(func(os.path.join(dirin,eachfile),*args))
    return result_list

class pair_ID:
    DIP_Ecoli = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_TMP_nonTMP/Ecoli_TMP_nonTMP_1156.txt'
class Feature_DB:
    DIP = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_featuredb/'
    IMEx = '/home/jjhnenu/data/PPI/release/featuredb/seq_feature_1D/'
    PANs = '/home/jjhnenu/data/PPI/release/otherdata/pans/featuredb/'
class Feature_pair_DB:
    DIP = '/home/jjhnenu/data/PPI/release/otherdata/DIP/_feature/'
    IMEx0 = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/0/all/'
    IMEx1 = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/1/all/'
    IMEx2 = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/2/all/'
    IMEx3 = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/3/all/'
    IMEx4 = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/4/all/'