# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/1/2 12:34
@desc:跨膜蛋白，就用KW-0812 Transmembrane这个列表
大部分方法没有考虑蛋白质的曾用名，列表不准确
"""
import os
import re


# 从列表txt中读取ID列表--common
from common import saveList, readIDlist, countpair, handledir, check_path


def getTmp_SpPair(tmpf,spf,finPair,foutPair,type1='TMP',type2='SP',crossover = False):
    """
    cretira:/home/jjhnenu/data/PPI/release/criteria/    20200701
    tmpf='/home/jjhnenu/data/PPI/release/criteria/allcession_KW-0812_131609.list'
    spf='/home/jjhnenu/data/PPI/release/criteria/allcession_soluble_614454.list'
    :param tmpf:
    :param spf:
    :param finPair:
    :param foutPair:
    :return:
    tmpf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\tmp\\KW-0812.list'
    spf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\sp\\splist.list'
    finPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\noprepeat.txt'
    foutPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\tmp_sp\\tmp_sp.txt'
    getTmp_SpPair(tmpf, spf, finPair, foutPair)
    """

    tmplist = readIDlist(tmpf)
    splist = readIDlist(spf)
    print(len(tmplist),len(splist))
    # len(tmplist)<len(splist)
    splitmark = '\\'
    if '/' in foutPair:
        splitmark = '/'
    mypath = foutPair[:foutPair.rindex(splitmark)+1]
    fSP_SP = mypath+'%s_%s.txt'%(type2,type2)
    fTMP_TMP = mypath+'%s_%s.txt'%(type1,type1)

    fnonTMP_SP = mypath + 'non%s_%s.txt' % (type1, type2)
    fTmp_nonSP = mypath + '%s_non%s.txt' % (type1, type2)

    fTMP_nonTmp = mypath+'%s_non%s.txt'%(type1,type1)
    fnonTmp_nonTmp = mypath+'non%s_non%s.txt'%(type1,type1)

    fnonsp_sp = mypath+'non%s_%s.txt'%(type2,type2)
    fnonsp_nonsp = mypath+'non%s_non%s.txt'%(type2,type2)

    fdrop = mypath+'drop.txt'
    with open(finPair, 'r') as fin, \
            open(foutPair, 'w') as fout,\
            open(fSP_SP,'w') as foutsp,\
            open(fTMP_TMP,'w') as fouttmp,\
            open(fnonTMP_SP,'w') as foutnonTMP_SP,\
            open(fTmp_nonSP,'w') as foutTmp_nonSP,\
            open(fTMP_nonTmp,'w') as foutTMP_nonTmp,\
            open(fnonTmp_nonTmp,'w') as foutnonTmp_nonTmp,\
            open(fnonsp_sp, 'w') as fout_nonsp_sp,\
            open(fnonsp_nonsp, 'w') as fout_nonsp_nonsp,\
            open(fdrop, 'w') as fout_drop\
            :
        line = fin.readline()
        while (line):
            pair = line.split('\t')
            try:
                a = pair[0]
                b = pair[1][:-1]
            except:
                print(pair)
            # Atmp = a in tmplist
            # Asp = False if Atmp and not crossover else a in splist
            # Btmp = b in tmplist
            # Bsp = False if Btmp and not crossover else b in splist

            Atmp = a in tmplist
            Asp =  a in splist
            Btmp = b in tmplist
            Bsp =  b in splist

            if Atmp and Bsp:    # TMP_SP
                fout.write(a + '\t' + b + '\n')
                fout.flush()
                print('%s %s save this pair' % (a, b))
            elif Asp and Btmp:  # TMP_SP
                fout.write(b + '\t' + a + '\n')
                fout.flush()
                print('%s %s save this pair' % (a, b))
            elif Asp and Bsp:   # SP_SP
                foutsp.write(a + '\t' + b + '\n')
                foutsp.flush()
                print('%s %s save this pair %s_%s' % (a, b,type2,type2))
            elif Atmp and Btmp: # TMP_TMP
                fouttmp.write(a + '\t' + b + '\n')
                fouttmp.flush()
                print('%s\t%s save this pair %s_%s' % (a, b,type1,type1))
            else:
                pass

            if not Atmp and Bsp:
                foutnonTMP_SP.write(a + '\t' + b + '\n')
                foutnonTMP_SP.flush()
                print('%s %s save this pair %s_non%s' % (a, b, type1, type2))
            elif Atmp and not Bsp:
                foutTmp_nonSP.write(a + '\t' + b + '\n')
                foutTmp_nonSP.flush()
                print('%s %s save this pair %s_non%s' % (a, b, type1, type2))
            else:pass

            # if (a in tmplist and b not in tmplist) or (a not in tmplist and b in tmplist):
            #     foutTMP_nonTmp.write(a + '\t' + b + '\n')
            #     foutTMP_nonTmp.flush()
            #     print('%s %s save this pair TMP_nonTmp' % (a, b))
            # if a not in tmplist and b not in tmplist:
            #     foutnonTmp_nonTmp.write(a + '\t' + b + '\n')
            #     foutnonTmp_nonTmp.flush()
            #     print('%s %s save this pair nonTMP_nonTmp' % (a, b))
            if Atmp and not Btmp:   # TMP_nonTMP
                foutTMP_nonTmp.write(a + '\t' + b + '\n')
                foutTMP_nonTmp.flush()
                print('%s %s save this pair %s_non%s' % (a, b,type1,type1))
            elif not Atmp and Btmp: # TMP_nonTMP
                foutTMP_nonTmp.write(b + '\t' + a + '\n')
                foutTMP_nonTmp.flush()
                print('%s %s save this pair %s_non%s' % (a, b,type1,type1))
            elif not Atmp and not Btmp:
                foutnonTmp_nonTmp.write(a + '\t' + b + '\n')
                foutnonTmp_nonTmp.flush()
                print('%s %s save this pair non%s_non%s' % (a, b,type1,type1))
            else:
                pass
            if not Asp and Bsp:
                fout_nonsp_sp.write(a + '\t' + b + '\n')
                fout_nonsp_sp.flush()
                print('%s %s save this pair non%s_%s' % (a, b,type2,type2))
            elif Asp and not Bsp:
                fout_nonsp_sp.write(b + '\t' + a + '\n')
                fout_nonsp_sp.flush()
                print('%s %s save this pair non%s_%s' % (a, b,type2,type2))
            elif not Asp and not Bsp:
                fout_nonsp_nonsp.write(a + '\t' + b + '\n')
                fout_nonsp_nonsp.flush()
                print('%s %s save this pair non%s_non%s' % (a, b,type2,type2))
            else:
                # fout_drop.write(a + '\t' + b + '\n')
                # print('%s %s drop this pair' % (a, b))
                pass

            line = fin.readline()
    func = countpair
    handledir(mypath, func)
    print('get%s_%sPair end'%(type1,type2))
def pairToproteinList(fin,fout=''):
    """
    # Gets a list of non-repeating proteins from the protein pair
    :param fin:
    :param fout:
    :return:
    """
    proteinlist = []
    with open(fin,'r') as fi:
        line = fi.readline()
        while(line):
            pair = line.split('\t')
            a = pair[0]
            b = pair[1][:-1]
            if a not in proteinlist:
                proteinlist.append(a)
                # fo.write(a+'\n')
                # fo.flush()
            if b not in proteinlist:
                proteinlist.append(b)
                # fo.write(b+'\n')
                # fo.flush()
            line = fi.readline()
    if fout:saveList(proteinlist,fout)
    print('pairToproteinList end')
    return proteinlist
def dictToFasta(fin,fout):
    mydict = loadDict(fin)
    with open(fout,'w')as fo:
        for key in mydict.keys():
            fo.write('>%s\n'%key)
            fo.write(mydict[key]+'\n')
            fo.flush()
def loadDict(fin):
    with open(fin,'r')as fo:
        mydict = fo.readline()
        return eval(mydict)
def getFastaSeq(fin):
    mydict = {}
    from Bio import SeqIO
    for record in SeqIO.parse(fin, 'fasta'):
        ID = record.id.split('|')[1]
        seq = record.seq
        mydict[ID]=str(seq)
    return mydict

if __name__ == '__main__':
    print('test')
    '''
    getSP

    '''
    # kwf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\sp\KW'
    # spf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\sp\\Swissprot.list'
    # fout = spf[:spf.rindex('\\')]+'\\splist.list'
    # getSP(kwf, spf, fout)
    '''
    getTmp_SpPair
    '''
    # tmpf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\tmp\\KW-0812.list'
    # # tmpf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\sp\\Swissprot.list'
    # spf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\sp\\splist.list'
    # finPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\noprepeat.txt'
    # # finPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\protein2960fortest.txt'
    # foutPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\tmp_sp\\tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)

    # fin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\combine_norepeat.txt'
    # fout = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\combine_norepeat_allprotein.txt'
    # pairToproteinList(fin, fout)

    # allProteinFilePath = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\combine_norepeat_allprotein.txt'
    # uniprotTMPFilePath = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\tmp\\uniprot-KW-0812.list'
    # tmpListFilePath = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\intact_tmp.txt'
    # nonTmpListFilePath = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\intact_nonTmp.txt'
    # classifyTAndN(allProteinFilePath, uniprotTMPFilePath, tmpListFilePath, nonTmpListFilePath)
    '''
    find tmp in intact_pair combine_norepeat_allprotein
    '''
    # fin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\combine_norepeat_allprotein.txt'
    # fCriter = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\tmp\\uniprot-KW-0812.list'
    # fyes = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\intact_tmp.txt'
    # classifyWithLargeCriterion(fin, fCriter, fyes)
    '''
    find tmp_nontmp pair
    get 5w pair for 28w pair
    '''
    # tmpf = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\intact_tmp.txt'
    # finPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\\combine_norepeat.txt'
    # outdir = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN'
    # checkPair(tmpf, finPair, outdir)
    '''
    find nontmp in 5w pair
    1.get nontmp
    2.drop repeat
    '''
    # finPair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\tmp_nonTmp.txt'
    # fout = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\nontmp.txt'
    # getCol(finPair, fout, col=1)
    '''
    nontmp exclude from five list ==> sp protein list in intact 
    '''
    # fin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\nontmp.txt'
    # dirin = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\uniprot'
    # fout = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\interactor_interaction\classifyTandN\\intact_sp.txt'
    # exclusiveFromMultipleF(fin, dirin, fout)

    '''
    SC
    '''
    # tmpf = '/home/jjhnenu/data/PPI/stage2/all_accession/KW/KW-0812.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/S_C/3Scere20170205_id_pair.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/S_C/tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)
    '''
    Ecoli
    '''
    # tmpf = '/home/jjhnenu/data/PPI/stage2/all_accession/KW/KW-0812.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/2Ecoli20170205_id_pair_12246.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)
    '''
    Human
    '''
    # tmpf = '/home/jjhnenu/data/PPI/stage2/all_accession/KW/KW-0812.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/Human/Hsapi20170205_id_pair_7417.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/Human/tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)
    '''
    Mus
    '''
    # tmpf = '/home/jjhnenu/data/PPI/stage2/all_accession/KW/KW-0812.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/Mus/Mmusc20170205_id_pair_2495.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/Mus/tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)
    '''
    HP
    '''
    # tmpf = '/home/jjhnenu/data/PPI/stage2/all_accession/KW/KW-0812.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/HP/Hpylo20170205_id_pair_1372.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/HP/tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)

    '''
    uniprotPairFromImex
    '''
    # tmpf = '/home/jjhnenu/data/PPI/stage2/all_accession/KW/KW-0812.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist_all_accession614463.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/uniprotPairFromImex.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)
    '''
    positive_2049 tmp_sp
    '''
    # tmpf = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0812_131609.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist_all_accession614463.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/uniprotPairFromImex.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/tmp_sp.txt'
    # getTmp_SpPair(tmpf, spf, finPair, foutPair)
    '''
    positive 2049 tmp_mp
    '''
    # tmpf = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0812_131609.list'
    # mpf = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0472_189043.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/uniprotPairFromImex.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/mp/tmp_mp.txt'
    # getTmp_SpPair(tmpf, mpf, finPair, foutPair)

    '''
    mp_nonmp
    '''
    # mpf = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0472_189043.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist_all_accession614463.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/uniprotPairFromImex.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/mp_sp/MP_SP.txt'
    # _path,_fname = os.path.split(foutPair)
    # check_path(_path)
    # getTmp_SpPair(mpf, spf, finPair, foutPair,type1='MP',type2='SP',crossover = False)

    '''
    MP_SP3859
    '''
    # tmpf = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0812_131609.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist_all_accession614463.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/mp_sp/MP_SP.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPairFromImex/mp_sp/MP_SP3859/TMP_SP.txt'
    # _path,_fname = os.path.split(foutPair)
    # check_path(_path)
    # getTmp_SpPair(tmpf, spf, finPair, foutPair,type1='TMP',type2='SP',crossover = False)

    '''
    other data
    '''
    # mpf = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0472_189043.list'
    # tmpf = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0812_131609.list'
    # spf = '/home/jjhnenu/data/PPI/sp/splist_all_accession614463.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/2Ecoli20170205_id_pair_12246.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/TMP_SP/TMP_SP.txt'
    # _path,_fname = os.path.split(foutPair)
    # check_path(_path)
    # getTmp_SpPair(tmpf, spf, finPair, foutPair,type1='TMP',type2='SP',crossover = True)
    '''
    splist allaccession
    '''
    # fin = '/home/jjhnenu/data/PPI/sp/not_KW_0472_448049.list'
    # fout = '/home/jjhnenu/data/PPI/sp/not_KW_0472_all_accession.list'
    # fnotInDb = '/home/jjhnenu/data/PPI/sp/not_KW_0472_notinsw.list'
    # getAllaccession(fin, fout,fnotInDb)
    '''
    mp allaccession
    '''
    # fin = '/home/jjhnenu/data/PPI/sp/KW/KW-0472.list'
    # fout = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0472_origin.list'
    # fnotInDb = '/home/jjhnenu/data/PPI/sp/KW/notinsw_KW-0472.list'
    # getAllaccession(fin, fout,fnotInDb)
    '''
    tmp allaccession
    '''
    # fin = '/home/jjhnenu/data/PPI/sp/KW/KW-0812.list'
    # fout = '/home/jjhnenu/data/PPI/sp/KW/allcession_KW-0812_origin.list'
    # fnotInDb = '/home/jjhnenu/data/PPI/sp/KW/notinsw_KW-0812.list'
    # getAllaccession(fin, fout,fnotInDb)


    '''
    count pair
    '''
    # dirin = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/TMP_SP'
    # func=countpair
    # handledir(dirin,func)


    '''get sp list'''
    # kwf = '/home/jjhnenu/data/PPI/sp/KW/KW-0472.list'
    # spf = '/home/jjhnenu/data/PPI/sp/Swissprot.list'
    # fout = os.path.join('/home/jjhnenu/data/PPI/sp/soluble','not0472_soluble.list')
    # getSP(kwf, spf, fout)

    '''
    other data
    
    Ecoli
    HP
    Human
    Mus
    SC
    uniprotPairFromImex
    '''
    # # mpf = '/home/jjhnenu/data/PPI/release/criteria/allcession_KW-0472_189043.list'
    # tmpf = '/home/jjhnenu/data/PPI/release/criteria/allcession_KW-0812_131609.list'
    # spf = '/home/jjhnenu/data/PPI/release/criteria/allcession_soluble_614454.list'
    # finPair = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/2Ecoli20170205_id_pair_12246.txt'
    # foutPair = '/home/jjhnenu/data/PPI/release/otherdata/Ecoli/TMP_SP/TMP_SP.txt'
    # _path,_fname = os.path.split(foutPair)
    # check_path(_path)
    # getTmp_SpPair(tmpf,spf,finPair, foutPair,type1='TMP',type2='SP',crossover = True)

    '''
    count pair
    '''
    # dirout_pair = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\otherdata\DIP\_TMP_nonTMP_qualified_drop_positive'
    # check_path(dirout_pair)
    # for eachfile in os.listdir(dirout_pair):
    #     fout_pair = os.path.join(dirout_pair, eachfile)
    #     countpair(fout_pair)
    '''
    imex 2020 0708
    '''

    tmpf = '/home/jjhnenu/data/PPI/release/criteria/allcession_KW-0812_131609.list'
    spf = '/home/jjhnenu/data/PPI/release/criteria/allcession_soluble_614454.list'
    finPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPiarFromImex20200709/getpair15955.csv'
    foutPair = '/home/jjhnenu/data/PPI/release/otherdata/uniprotPiarFromImex20200709/TMP_SP/TMP_SP.txt'
    _path,_fname = os.path.split(foutPair)
    check_path(_path)
    getTmp_SpPair(tmpf,spf,finPair, foutPair,type1='TMP',type2='SP',crossover = True)