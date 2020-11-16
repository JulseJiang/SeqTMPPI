# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/7/22 15:23
@desc:
"""
import os

from keras import Sequential, models
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense
import tensorflow as tf
import keras.backend as K
from myData import BaseData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def check_path(in_dir):
    if '.' in in_dir:
        in_dir,_ = os.path.split(in_dir)
    if not os.path.exists(in_dir):os.makedirs(in_dir)

def metric_ACC(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    acc = (TP + TN) / (TP + FP + TN + FN)
    return acc
#精确率评价指标
def metric_precision(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    return precision
#召回率评价指标
def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN)
    return recall
#F1-score评价指标
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score
# MCC
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
def evaluate_manual(y_true,y_pred):
    y_true = tf.constant(list(y_true), dtype=float)
    y_pred = tf.constant(list(y_pred),dtype=float)
    return [K.eval(metric_ACC(y_true, y_pred)),
            K.eval(metric_precision(y_true, y_pred)),
            K.eval(metric_recall(y_true, y_pred)),
            K.eval(metric_F1score(y_true, y_pred)),
            K.eval(matthews_correlation(y_true, y_pred))]
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

'''
input
'''
idx = 1
fin_dir = '/home/jjhnenu/data/PPI/release/pairdata/p_fw_v1/%d/0/'% idx
dir_in = '/home/jjhnenu/data/PPI/release/feature/p_fp_fw_19471' # feature

fin_pair = '%s/train.txt' % fin_dir
fin_validate = '%s/validate.txt' % fin_dir
fin_test = '%s/test.txt' % fin_dir


dirout = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive_test/%d' % idx
check_path(dirout)

print(dirout)


'''
load dataset
'''
onehot = True
bd = BaseData()
x_train, y_train= bd.loadTest(fin_pair,dir_in,onehot=onehot)
x_validate, y_validate = bd.loadTest(fin_validate,dir_in,onehot=onehot)
x_test, y_test = bd.loadTest(fin_test,dir_in,onehot=onehot)
'''
param
'''
input_shape = x_train.shape[1:]
kernel_size = 99
epochs=80
filters = 300
batch_size = 90
metrics = ['acc',metric_precision, metric_recall, metric_F1score, matthews_correlation]
# metrics = ['binary_accuracy', 'categorical_accuracy',metric_precision, metric_recall, metric_F1score, matthews_correlation]
metric_json = {
        'metric_precision': metric_precision,
        'metric_recall': metric_recall,
        'metric_F1score': metric_F1score,
        'matthews_correlation': matthews_correlation
    }
'''
build model
'''
model = Sequential()
model.add(
    Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=metrics)
model.summary()
'''
fit model
'''
history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_validate, y_validate))
'''
save model and training result
'''
plot_result(history.history,dirout)
model.save(os.path.join(dirout,'_my_model.h5'))  # creates a HDF5 file 'my_model.h5'
json_string = model.to_json()
with open(os.path.join(dirout,'_my_model.json'), 'w') as fo:
    fo.write(json_string)
    fo.flush()

history_dict = history.history
with open(os.path.join(dirout,'_history_dict.txt'), 'w') as fo:
    fo.write(str(history_dict))
    fo.flush()
with open(os.path.join(dirout , '_evaluate.txt'), 'w') as fi:
    fi.write('evaluate on validate dataset:' + str(model.evaluate(x_validate, y_validate, verbose=False,batch_size=90)) + '\n')
    fi.write('evaluate on testdataset:' + str(model.evaluate(x_test, y_test, verbose=False, batch_size=90)) + '\n')
    fi.write('evaluate on train dataset:' + str(model.evaluate(x_train, y_train, verbose=False, batch_size=90)) + '\n')
    fi.write('history.params:' + str(history.params) + '\n')
'''
save manual predict result
'''
result_predict = model.predict(x_test, batch_size=90)
result_predict = result_predict.reshape(-1)
result_class = model.predict_classes(x_test, batch_size=90)
result_class = result_class.reshape(-1)
result_manual = evaluate_manual(y_test, result_predict)

df = pd.read_table(fin_test,header=None)
df.columns = ['tmp', 'nontmp', 'real_label']
df['predict_label'] = result_class
df['predict'] = result_predict
df.to_csv(os.path.join(dirout,'y_predict.csv'),index=False)

with open(os.path.join(dirout , '_evaluate_predict.txt'), 'w') as fj:
    fj.write(str(result_manual))
    fj.write(str(metrics))
'''
reload model and show result
'''

model_1 = models.load_model(os.path.join(dirout,'_my_model.h5'), custom_objects=metric_json)
result_1 = model_1.evaluate(x_test, y_test, verbose=False,batch_size=batch_size)
result_predict_1 = model_1.predict(x_test,batch_size=batch_size)
result_class_1 = model_1.predict_classes(x_test,batch_size=batch_size)
result_manual_1 = evaluate_manual(y_test, result_predict_1)

df['predict_label_1'] = result_class_1
df['predict_1'] = result_predict_1
df.to_csv(os.path.join(dirout,'y_predict_reload.csv'),index=False)

with open(os.path.join(dirout , '_evaluate_predict_reload.txt'), 'w') as fk:
    fk.write(str(result_manual))
    fk.write(str(metrics))

# training
# positive :  1652
# negative :  1626

# validate
# positive :  204
# negative :  205

# test
# positive :  192
# negative :  217

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv1d_1 (Conv1D)            (None, 3902, 300)         624000
# _________________________________________________________________
# global_average_pooling1d_1 ( (None, 300)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 301
# =================================================================
# Total params: 624,301
# Trainable params: 624,301
# Non-trainable params: 0
# _________________________________________________________________
# Train on 3278 samples, validate on 409 samples

# val_loss,0.4176876719831546
# val_acc,0.8044009804725647
# val_metric_precision,0.8069379925727844
# val_metric_recall,0.8049325942993164
# val_metric_F1score,0.8052012324333191
# val_matthews_correlation,0.6008391976356506
# loss,0.36541583866808314
# acc,0.8395363
# metric_precision,0.86527026
# metric_recall,0.8137184
# metric_F1score,0.8331276
# matthews_correlation,0.68184453

# ['loss','acc',metric_precision, metric_recall, metric_F1score, matthews_correlation]
# model.evaluate(x_validate, y_validate, verbose=False,batch_size=90)
# [0.4176876719831546, 0.8044009804725647, 0.8069379925727844, 0.8049325942993164, 0.8052012324333191, 0.6008391976356506]


# ['acc',metric_precision, metric_recall, metric_F1score, matthews_correlation]
# train_result = evaluate_manual(y_train, model.predict(x_train, batch_size=batch_size))
# [0.4999516, 0.50396585, 0.49389872, 0.4988815, 0.0]

# validate_result = evaluate_manual(y_validate, model.predict(x_validate, batch_size=batch_size))
# [0.500003, 0.4987775, 0.4987775, 0.4987775, 0.0]

# validate_result = evaluate_manual(y_validate, model.predict(x_validate, batch_size=409))
# [0.500003, 0.4987775, 0.4987775, 0.4987775, 0.0]

# result_manual
# # [0.5018681, 0.46943766, 0.46943766, 0.46943763, 0.0]

# result_manual_1
# [0.5018681, 0.46943766, 0.46943766, 0.46943763, 0.0]

# train_result = evaluate_manual(y_train, model_1.predict(x_train, batch_size=90))
# [0.4999516, 0.50396585, 0.49389872, 0.4988815, 0.0]

# result_class = result_class.reshape(-1)
