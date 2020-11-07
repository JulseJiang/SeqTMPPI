# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/17 17:52
@desc:
"""

class Protein:
    def checkProtein(self,seq, min, max,uncomm=True):
        if uncomm:
            return self.checkUncomm(seq) and self.checkLength(seq, min=min, max=max)
        else:return self.checkLength(seq, min=min, max=max)

    def checkUncomm(self,seq):
        '''
        if protein seq contains uncommon amino acid,return False
        :param seq:
        :return:
        '''
        uncommList = list('XOUBZJ')
        for u in uncommList:
            if u in seq:
                print('protein seq contains uncommon amino acid', u)
                return False
        return True

    def checkLength(self,seq, min=50, max=2000):
        length = len(seq)
        if min == max: return True  # no checking
        if length >= min and length <= max:
            return True
        print('%s length of protein seq not [50-2000]' % length)
        return False


