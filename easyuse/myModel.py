# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/15 20:14
@desc:
"""
import os
from functools import wraps
from keras import models, Input, Model, callbacks
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, Conv2D, Conv1D, Embedding, GlobalAveragePooling2D, Flatten, \
    Dropout, Activation, MaxPooling2D, MaxPooling1D, LSTM, concatenate, Concatenate, GlobalMaxPooling1D
# from keras.utils import plot_model

from myEvaluate import MyEvaluate
from mySupport import plot_result


class Param:
    metrics=MyEvaluate.metric

    CNN1D = 'CNN1D'
    CNN1D_OH = 'CNN1D_OH' # onehot
    CNN2D = 'CNN2D'
    LSTM = 'LSTM'
    DNN = 'DNN'
    CNN_LSTM = 'CNN_LSTM'
    CNN1D_MAX_OH = 'CNN1D_MAX_OH'

class MyModel(object):
    def __init__(self,
                input_shape = (160,),
                filters = 250,
                kernel_size = 3,
                pool_size = 2,
                hidden_dims = 250,
                batch_size=100,
                epochs = 60,
                metrics = None,
                model_type = Param.CNN1D
                 ):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_type = model_type
        if metrics == None:self.metrics = Param.metrics
    def __call__(self, func,x_train,y_train,validation_data=None,*args, **kwargs):
        @wraps(func) # todo
        def wrapper(x_train,y_train,validation_data=validation_data,*args, **kwargs):
            # history = self.process(x_train,y_train,validation_data=validation_data)
            # return func(history, *args, **kwargs)
            pass
        return wrapper

    def process(self,fout,x_train, y_train, x_test,y_test):
        if self.model_type == Param.CNN_LSTM:
            fixlen = int(self.input_shape[0]/2)
            x_train = [x_train[:,:fixlen],x_train[:,fixlen:]]
            x_test = [x_test[:,:fixlen],x_test[:,fixlen:]]
        print('x_train.shape,x_test.shape',x_train[0].shape,x_test[0].shape)
        self.loadModel()
        self.complie()
        self.fit(x_train,y_train,validation_data=(x_test, y_test))
        self.save_model(fout)
        self.save_result(fout,x_test,y_test)
        # plot_result(self.history.history,fout)
    def process_re_emerge(self,fin_model,x_test,y_test):
        self.loadExistModel(fin_model)
        print(self.evaluate(x_test, y_test))

    # support process
    def loadModel(self):
        model =None
        if self.model_type==Param.CNN1D:model =self.CNN1D()
        elif self.model_type==Param.CNN1D_OH:model =self.CNN1D_OH()
        elif self.model_type == Param.CNN1D_MAX_OH:model = self.CNN1D_MAX_OH()
        elif self.model_type==Param.CNN2D:model =self.CNN2D()
        elif self.model_type==Param.LSTM:model =self.LSTM()
        elif self.model_type==Param.DNN:model =self.DNN()
        elif self.model_type == Param.CNN_LSTM:model = self.CNN_LSTM()
        else:assert 'no such model'
        self.model = model

    def complie(self):
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=self.metrics)
        self.model.summary()
    def fit(self,x_train,y_train,validation_data=None):
        """
        :param x_train:
        :param y_train:
        :param validation_data: validation_data=(x_test, y_test)
        :return:history
        """
        self.history  = self.model.fit(x_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=validation_data,
                            # callbacks=[EarlyStopping(monitor='loss', patience=3)]
                                       )

    def save_model(self,fout):
        # plot_model(self.model, to_file=os.path.join(fout,'_model.png'), show_shapes=True, show_layer_names='False', rankdir='TB')
        self.model.save(os.path.join(fout,'_my_model.h5'))  # creates a HDF5 file 'my_model.h5'
        json_string = self.model.to_json()
        with open(os.path.join(fout,'_my_model.json'), 'w') as fo:
            fo.write(json_string)
            fo.flush()

    def save_result(self,fout,x_test,y_test):
        history_dict = self.history.history
        with open(os.path.join(fout,'_history_dict.txt'), 'w') as fo:
            fo.write(str(history_dict))
            fo.flush()
        with open(os.path.join(fout , '_evaluate.txt'), 'w') as fi:
            fi.write('evaluate:' + str(self.evaluate(x_test, y_test)) + '\n')
            fi.write('history.params:' + str(self.history.params) + '\n')

    def CNN1D(self):
        model = Sequential()
        model.add(Embedding(21, 160, input_shape=self.input_shape))
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model
    def CNN1D_OH(self):
        model = Sequential()
        model.add(
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shape))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def CNN1D_MAX_OH(self):
        model = Sequential()
        model.add(
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D())
        # model.add(Flatten())
        # model.add(
        #     Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def CNN2D(self):
        model = Sequential()
        model.add(Conv2D(32, self.kernel_size, padding='valid',
                         input_shape=self.input_shape, data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, self.kernel_size, padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='sigmoid'))

        return model


    def LSTM(self):
        pass

    def DNN(self):
        pass

    def CNN_LSTM(self):
        # main_input_a = Input(shape = self.input_shape/2)
        _shape = (int(self.input_shape[0]/2),)
        print('_shape',_shape)

        input_a = Input(shape=_shape)
        embedding_a = Embedding(21, 128)(input_a)
        conv1d_a = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(embedding_a)
        pool_a = MaxPooling1D()(conv1d_a)
        lstm_a = LSTM(80)(pool_a)


        input_b = Input(shape=_shape)
        embedding_b = Embedding(21, 128)(input_b)
        conv1d_b = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(embedding_b)
        pool_b = MaxPooling1D()(conv1d_b)
        lstm_b = LSTM(80)(pool_b)

        concat = concatenate([lstm_a,lstm_b],axis=-1)
        predictions = Dense(1, activation='sigmoid')(concat)

        model = Model(inputs=[input_a,input_b],outputs=predictions)
        model.evaluate()
        return model

    # support process_re_emerge
    def loadExistModel(self,fin_model):
        self.model = models.load_model(
            fin_model,
            custom_objects=MyEvaluate.metric_json)
        self.model.summary()
    # support save_result
    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test, y_test, verbose=False,batch_size=self.batch_size)

# class testModel():
    # fin_pair = '/home/jjhnenu/data/PPI/release/pairdata/group/p_fp_1_1/0/all.txt'
    # dir_in = '/home/jjhnenu/data/PPI/release/feature/group/p_fp_1_1/0/all/'
    # fin_model = '/home/19jiangjh/data/PPI/release/result_in_paper/alter_param/alter_k_99_f300_b90/9/1/_my_model.h5'
    #
    # model = models.load_model(fin_model)
    # model.evaluate(x_test, y_test, verbose=False)


