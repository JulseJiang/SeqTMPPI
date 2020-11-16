# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:25:24 2020

@author: YY
"""
import math
import sys

from sklearn.metrics import roc_curve,auc,average_precision_score

def calculate_performance(test_num,labels,predict_y,predict_score):
	tp=0
	fp=0
	tn=0
	fn=0
	for index in range(test_num):
		if(labels[index]==1):
			if(labels[index] == predict_y[index]):
				tp += 1
			else:
				fn += 1
		else:
			if(labels[index] == predict_y[index]):
				tn += 1
			else:
				fp += 1
	acc = float(tp+tn)/test_num
	precision = float(tp)/(tp+fp+ sys.float_info.epsilon)
	sensitivity = float(tp)/(tp+fn+ sys.float_info.epsilon)
	specificity = float(tn)/(tn+fp+ sys.float_info.epsilon)
	f1 = 2 * precision * sensitivity / (precision + sensitivity + sys.float_info.epsilon)
	mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + sys.float_info.epsilon)
	# mcc = float(tp*tn-fp*fn)/(np.sqrt(int((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
	aps = average_precision_score(labels,predict_score)
	fpr,tpr,_ = roc_curve(labels,predict_score)
	aucResults = auc(fpr,tpr)
	return [acc,precision,sensitivity,f1,mcc,aps,aucResults,specificity]
if __name__ == '__main__':
	import pandas as pd
	# result_out = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/1/validate/result.csv'
	# result_out = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\result_in_paper\alter_ratio\p_fw_v1_train_validate_v2_fixpositive\1\validate\result.csv'
	result_out = r'E:\githubCode\BioComputeCodeStore\JiangJiuhong\data\PPI\release\result_in_paper\alter_ratio\p_fw_v1_train_validate_v2_fixpositive\10\train\result.csv'
	df = pd.read_csv(result_out)
	y_true = df['real_label']
	y_pred = df['predict']
	y_pred_label = df['predict_label']

	test_num = len(y_true)
	labels = list(y_true)
	predict_y = y_pred_label
	predict_score = y_pred

	myresult = calculate_performance(test_num, labels, predict_y, predict_score)
	print('calculate_performance:\t [acc,precision,sensitivity,f1,mcc,aps,aucResults,specificity] \n',myresult)


