from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Bidirectional, Input, Flatten, Activation, Reshape, RepeatVector, Concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import LearningRateScheduler
import math
import matplotlib.pyplot as plt
import pickle



from keras.callbacks import EarlyStopping

import tensorflow as tf
import os
import time
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import time

import time
import os
import sys
import pandas as pd
import numpy as np
from DataLoader import *
# import os


LOAD_DATASET=False
datatset_str="multi_sub_shuffled_"



os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

use_lr_scheduler = True
lr_factor = 0.5
lr_patience = 8
lr_threshold = 0.01
lr_min_rate = 1e-6

path="/home/vtp/masters_proj/GaitPhase"

result_path= "/home/vtp/masters_proj/GaitPhase/Results"

#Learning rate scheduler
def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.005
	epochs_drop = 30
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

#loss 
class MyHuberLoss(Loss):
    # initialize instance attributes
    def __init__(self, threshold=0.06):
        super(MyHuberLoss, self).__init__()
        self.threshold = threshold
    
# Compute loss
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - self.threshold / 2)
        return tf.where(is_small_error, small_error_loss, big_error_loss)

class model:
    def __init__(self, model_num=1) -> None:
        self.epochs=30
        self.batch_size=32
        self.learning_rate=1e-3
        self.checkpoint_path=path+"cnn_lstm.ckpt"
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        self.model_save_path=path
        self.data=DataLoader(path)
        self.model_num=model_num


    def create_model(self):
        model1 = Sequential()
        model1.add(InputLayer((10, 11)))
        model1.add(Conv1D(filters=64, kernel_size=4,strides=1,
            activation='relu', padding='same'))
        # model1.add(Conv1D(filters=64, kernel_size=5,strides=1,
        #     activation='relu'))
        model1.add(MaxPooling1D(pool_size=2))
        # model1.add(Flatten())
        # model1.add(RepeatVector(FORECAST_RANGE))
        model1.add(LSTM(100, return_sequences=True))
        model1.add(LSTM(64, return_sequences=True))
        # model1.add(LSTM(8))
        # model1.add(TimeDistributed(Dense(50, activation='relu')))
        # model1.add(TimeDistributed(Dense(2)))
        # model1.add(LSTM(1, return_sequences=True))
        # model1.add(LSTM(64, return_sequences=True))

        # model1.add(LSTM(8))
        model1.add(Flatten())
        model1.add(Dense(16))
        model1.add(Dense(2))
        self.model=model1

    def pop_data(self, multi_sub=True):

        self.data_path=path+"/Subject_data/best_xls/"
        self.data.GetTrainTestData(self.data_path, multi_sub)
        # with open(self.data_path+ datatset_str+"train_x.pkl",'wb') as f:
        #     pickle.dump(self.data.train_x, f)
        # with open(self.data_path+ datatset_str+"train_y.pkl",'wb') as f:
        #     pickle.dump(self.data.train_y, f)
        # with open(self.data_path+ datatset_str+"test_x.pkl",'wb') as f:
        #     pickle.dump(self.data.validation_x, f)
        # with open(self.data_path+ datatset_str+"test_y.pkl",'wb') as f:
        #     pickle.dump(self.data.validation_y, f)

    def load_dataset(self):
        data_path=path+"/Subject_data/low_dim_data/"
        self.data.train_x=np.load(data_path+datatset_str+"train_x.pkl", allow_pickle=True)
        self.data.train_y=np.load(data_path+datatset_str+"train_y.pkl", allow_pickle=True)
        self.data.validation_x=np.load(data_path+datatset_str+"test_x.pkl", allow_pickle=True)
        self.data.validation_y=np.load(data_path+datatset_str+"test_y.pkl", allow_pickle=True)
        print(self.data.train_x[0:100,:])


    def train_model(self, model_num=1):

        self.model_num=model_num
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps =10000,
            decay_rate=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        monitor = EarlyStopping(monitor='loss', min_delta=1e-5, patience=3, verbose=1, mode='auto')
        self.create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        if model_num==1:
            model1 = Sequential()
            model1.add(InputLayer((10, self.data.train_x.shape[2])))
            model1.add(Conv1D(filters=64, kernel_size=4,strides=1,
                activation='relu', padding='same'))
            # model1.add(Conv1D(filters=64, kernel_size=5,strides=1,
            #     activation='relu'))
            model1.add(MaxPooling1D(pool_size=2))
            # model1.add(Flatten())
            # model1.add(RepeatVector(FORECAST_RANGE))
            model1.add(LSTM(100, return_sequences=True))
            model1.add(LSTM(64, return_sequences=True))
            # model1.add(LSTM(8))
            # model1.add(TimeDistributed(Dense(50, activation='relu')))
            # model1.add(TimeDistributed(Dense(2)))
            # model1.add(LSTM(1, return_sequences=True))
            # model1.add(LSTM(64, return_sequences=True))

            # model1.add(LSTM(8))
            model1.add(Flatten())
            model1.add(Dense(16))
            model1.add(Dense(2))
            self.model=model1

        if model_num==2:
            model2 = Sequential()
            model2.add(InputLayer((10,7)))
            model2.add(Conv1D(filters=32, kernel_size=3,strides=1,
                activation='relu', padding='same'))
            
            model2.add(MaxPooling1D(pool_size=2))

            model2.add(LSTM(64, return_sequences=True))
            model2.add(LSTM(16, return_sequences=True))
            # model1.add(LSTM(8))
            # model1.add(TimeDistributed(Dense(50, activation='relu')))
            # model1.add(TimeDistributed(Dense(2)))
            # model1.add(LSTM(1, return_sequences=True))
            # model1.add(LSTM(64, return_sequences=True))

            # model1.add(LSTM(8))
            model2.add(Flatten())
            model2.add(Dense(8, activation='tanh'))
            model2.add(Dense(2, activation='tanh'))
            self.model=model2

        if model_num==3:
            model3 = Sequential()
            model3.add(InputLayer((10,self.data.train_x.shape[2])))
            model3.add(Conv1D(filters=32, kernel_size=3,strides=1,
                activation='relu', padding='same'))            
            model3.add(AveragePooling1D(pool_size=3))

            model3.add(LSTM(64, return_sequences=True))
            model3.add(LSTM(16, return_sequences=True))
            # model1.add(LSTM(8))
            # model1.add(TimeDistributed(Dense(50, activation='relu')))
            # model1.add(TimeDistributed(Dense(2)))
            # model1.add(LSTM(1, return_sequences=True))
            # model1.add(LSTM(64, return_sequences=True))

            # model1.add(LSTM(8))
            lrate = LearningRateScheduler(step_decay)
            callbacks_list = [lrate, monitor]
            model3.add(Flatten())
            model3.add(Dense(2))
            self.model=model3

        if model_num==4:
            model4 = Sequential()
            model4.add(InputLayer((10,self.data.train_x.shape[2])))
            model4.add(Conv1D(filters=32, kernel_size=3,strides=1,
                activation='relu', padding='same'))            
            model4.add(AveragePooling1D(pool_size=3))

            model4.add(LSTM(8, return_sequences=True))
            model4.add(LSTM(2, return_sequences=True))
            model4.add(Conv1D(filters=16, kernel_size=3,strides=1,
                activation='relu', padding='same'))            
            # model1.add(LSTM(8))
            model4.add(Flatten())
            model4.add(Dense(2))
            self.model=model4

        print(self.model.summary())
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[RootMeanSquaredError()])
        print("X shape",self.data.train_x.shape)
        print("Y shape",self.data.train_y.shape)

        history = self.model.fit(self.data.train_x, self.data.train_y, epochs=self.epochs, batch_size=self.batch_size, callbacks=monitor,verbose=1, validation_split=0.2)   

        self.model.save(self.model_save_path+"CNN_LSTM_"+str(model_num)+"_"+str(3))


    def test_model(self, load_model=False):
        if load_model:
            self.model=tf.keras.models.load_model(self.model_save_path+"CNN_LSTM_"+ str(self.model_num)+"_"+str(3))

        nan_indices = np.isnan(self.data.train_y)
        # Find the rows containing NaN values
        nan_rows = np.any(nan_indices, axis=1)
        # Remove the rows containing NaN values
        self.data.train_y = self.data.train_y[~nan_rows]

        trainPredict = self.model.predict(self.data.train_x)
        testPredict = self.model.predict(self.data.validation_x)


        trainScore = np.sqrt(mean_squared_error(self.data.train_y, trainPredict))
        print('Train Score: %.3f RMSE' % (trainScore))
        testScore = np.sqrt(mean_squared_error(self.data.validation_y, testPredict))
        print('Test Score: %.3f RMSE' % (testScore))            

        pred = np.zeros((len(testPredict),1))

        for iter in range(len(testPredict)):
            x = testPredict[iter][0]
            y = testPredict[iter][1]
            pred[iter] = ((math.atan2(y,x) + 2*math.pi) % (2*math.pi)) * (100 / (2*math.pi))
        

        actual = np.zeros((len(self.data.validation_y),1))

        for iter in range(len(self.data.validation_y)):
            x =self.data.validation_y[iter][0]
            y =self.data.validation_y[iter][1]
            actual[iter] = ((math.atan2(y,x) + 2*math.pi) % (2*math.pi)) * (100 / (2*math.pi))


        fig=plt.figure(figsize=(35,15))

        # plt.plot(actual2[:],'-',label='Actual', linewidth = 3)
        # plt.plot(predict[:],'.-',label='prediction', linewidth = 2)



        plt.plot(actual[50:2556],'.',label='Actual')
        plt.plot(pred[50:2556],'.',label='prediction')

        plt.legend()
        plt.title('LSTM Prediction - Full Data')
        #plt.title('Right Foot')
        #plt.ylabel('Angle')
        #plt.xlabel('Gait Cycle Percentage')
        plt.xlabel('Data Point')
        plt.ylabel('Percentage (%)')
        plt.savefig(result_path+"/gait_cycle_res"+str(self.model_num)+"3.png")


        error_thresh_list=[1, 2, 3, 4, 5]

        for err_thresh in error_thresh_list:
            correct=0
            for iter in range(len(actual)):
                if actual[iter]<2 or actual[iter]>98:
                    correct+=1
                    continue
                if (abs(actual[iter] - pred[iter]) <= err_thresh):
                            correct+=1
            print("Error Threshold ", err_thresh, "%   Precision: ", correct * 100/len(actual))

# train_cnnlstm=model()
# train_cnnlstm.load_dataset()
# train_cnnlstm.train_model()
# train_cnnlstm.test_model()