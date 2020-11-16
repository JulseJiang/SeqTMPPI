# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/15 21:50
@desc:
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split

from common import getPairs


class Feature:
    f_bio_feature_1D = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\tensorFlowTest\file\resource\trainPair_label5000v2.npy'

    # onehot
    f_seq_feature_1D = r''
    f_seq_feature_2D = r''

    dir_pssm_feature_2D = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\feature\pssm_feature_2D\\'
class seqFeatureData:
    def loadData(self,limit = 0):
        dataset = np.load(r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\tensorFlowTest\file\resource\trainPair_label5000v2.npy')
        row, col = dataset.shape
        np.random.shuffle(dataset)
        # split into input (X) and output (y) variables
        if limit!=0 and limit <row:
            dataset = dataset[:limit]
        x = dataset[:, :col - 1]
        y = dataset[:, col - 1]
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=0.1, random_state=123)
        return (x_train, y_train), (x_test, y_test)
class TmpPSSMData:
    def __init__(self):
        self.positive = []
        self.negative = []

    """
    handle PSSM data as feature
    """
    def load(self,fin_pair,dir_in,test_size=0.1, random_state=123,limit=0):
        # (positive, negative)
        self.loadPair(fin_pair,dir_in,limit=limit)
        return self.process(test_size=test_size, random_state=random_state)
    def loadPAN(self,dir_positive,dir_negative,test_size=0.1, random_state=123,limit=0):
        self.positive = self.loadFile(dir_positive,limit=limit)
        self.negative = self.loadFile(dir_negative,limit=limit)
        return self.process(test_size=test_size, random_state=random_state)


    def process(self,test_size=0.1, random_state=123):
        data = np.vstack([self.positive, self.negative])
        label = np.hstack([np.ones(len(self.positive)), np.zeros(len(self.negative))])
        index = [x for x in range(len(label))]
        np.random.shuffle(index)
        data = data[index]
        label = label[index]
        data = np.reshape(data, data.shape + (1,))
        x_train, x_test, y_train, y_test = \
            train_test_split(data, label, test_size=test_size, random_state=random_state)
        return (x_train, y_train), (x_test, y_test)
    def loadFile(self,dir_in,limit=0):
        mylist = []
        row = 0
        for eachfile in os.listdir(dir_in):
            mylist.append(np.load(os.path.join(dir_in,eachfile)))
            row = row + 1
            if limit!=0 and row ==limit:break
        return np.stack(mylist)
    def loadPair(self,fin_pair,dir_in,limit=0):
        positive = []
        negative = []
        for proteins in getPairs(fin_pair,title=False):
            print('str(proteins)-----------------',str(proteins))
            eachfile = os.path.join(dir_in,'%s_%s.npy'%(proteins[0],proteins[1]))
            try:
                elem = np.load(os.path.join(dir_in, eachfile))
                if proteins[2] == '1':positive.append(elem)
                else:
                    negative.append(elem)
            except:
                print('not find feature of this pair',str(proteins))

        if limit!=0 and limit<min(len(positive),len(negative)):positive,negative = positive[:limit],negative[:limit]
        print('positive : ',len(positive))
        print('negative : ',len(negative))

        positive = np.stack(positive)
        negative = np.stack(negative)
        self.positive = positive
        self.negative = negative
class BaseData:
    def __init__(self):
        self.positive = []
        self.negative = []

    def load(self,fin_pair,dir_in,test_size=0.1, random_state=123,limit=0,onehot=False):
        # (positive, negative)
        self.loadPair(fin_pair,dir_in,limit=limit)
        return self.process(test_size=test_size, random_state=random_state,onehot=onehot)
    def loadPAN(self,dir_positive,dir_negative,test_size=0.1, random_state=123,limit=0):
        self.positive = self.loadFile(dir_positive,limit=limit)
        self.negative = self.loadFile(dir_negative,limit=limit)
        return self.process(test_size=test_size, random_state=random_state)

    def process(self,test_size=0.1, random_state=123,onehot=False):
        data = np.vstack([self.positive, self.negative]) if self.negative!=[] else self.positive
        label = np.hstack([np.ones(len(self.positive)), np.zeros(len(self.negative))]) if self.negative!=[] else np.ones(len(self.positive))
        return self.subprocess(data,label,test_size=test_size, random_state=random_state,onehot=onehot)

    def subprocess(self,data,label,test_size=0.1, random_state=123,onehot=False,is_shuffle=True):
        if is_shuffle:
            index = [x for x in range(len(label))]
            np.random.shuffle(index)
            data = data[index]
            label = label[index]
        if onehot:
            from keras.utils import to_categorical
            data = to_categorical(data)
        else:
            if len(data.shape) > 2:  # for 2D
                data = np.reshape(data, data.shape + (1,))
        if test_size == 0:
            return data, label
        else:
            print('end of the process')
            x_train, x_test, y_train, y_test = \
                train_test_split(data, label, test_size=test_size, random_state=random_state)
            return (x_train, y_train), (x_test, y_test)
    def loadFile(self,dir_in,limit=0):
        mylist = []
        row = 0
        for eachfile in os.listdir(dir_in):
            mylist.append(np.load(os.path.join(dir_in,eachfile)))
            row = row + 1
            if limit!=0 and row ==limit:break
        return np.stack(mylist)
    def loadPair(self,fin_pair,dir_in,limit=0):
        positive = []
        negative = []
        row = 0
        for proteins in getPairs(fin_pair,title=False):
            eachfile = os.path.join(dir_in,'%s_%s.npy'%(proteins[0],proteins[1]))
            try:
                elem = np.load(os.path.join(dir_in, eachfile))
                # loading test dataset or positive dataset
                if len(proteins) < 3 or proteins[2] == '1':
                    positive.append(elem)
                    row = row + 1
                else:
                    negative.append(elem)
            except:
                print('not find feature of this pair',str(proteins))
        if limit!=0 and limit<min(len(positive),len(negative)):positive,negative = positive[:limit],negative[:limit]
        print('positive : ',len(positive))
        print('negative : ',len(negative))
        positive = np.stack(positive)
        negative = np.stack(negative) if negative!=[] else []
        self.positive = positive
        self.negative = negative

    def loadTest(self,fin_pair,dir_in,onehot=False,is_shuffle=False):
        """

        :param fin_pair:
        :param dir_in:
        :param limit:
        :param onehot:
        :return: data,label
        """
        x_test = []
        y_test = []
        for proteins in getPairs(fin_pair, title=False):
            eachfile = os.path.join(dir_in, '%s_%s.npy' % (proteins[0], proteins[1]))
            try:
                # elem = np.load(os.path.join(dir_in, eachfile))
                elem = np.load(eachfile)
                x_test.append(elem)
                # loading test dataset or positive dataset
                if len(proteins) < 3 or proteins[2] == '1':
                    y_test.append(1)
                else:
                    y_test.append(0)
            except:
                print('not find feature of this pair', str(proteins))
        data = np.array(x_test)
        label = np.array(y_test)
        return self.subprocess(data,label,test_size=0, random_state=123,onehot=onehot,is_shuffle=is_shuffle)

if __name__ == '__main__':
   pass






