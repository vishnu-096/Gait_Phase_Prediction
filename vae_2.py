import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import time
import os
import sys
import json
import pickle

import scipy as sp
from scipy import signal

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import metrics


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Bidirectional, Input, Flatten, Activation, Reshape, RepeatVector, Concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os, warnings, random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers, Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def encode_gait_percentage(df, name):
    percent = df[name].values.tolist()
    X = np.zeros((len(df[name]),1))
    Y = np.zeros((len(df[name]),1))
    for i in range (len(percent)):
        phi = percent[i] * 2 * math.pi / 100
        X[i] = math.cos(phi)
        Y[i] = math.sin(phi)
    df['X'] = X
    df['Y'] = Y
  
def convert_data(d_x, d_y, look_back = 1, fore_cast = 1):
    dataX = []
    dataY = []

    for i in range(look_back, len(d_x) - fore_cast):
        dataX.append(d_x[i - look_back: i])
        dataY.append(d_y[i + fore_cast,:])

    return np.array(dataX), np.array(dataY)

def get_train_data_from_df(all_data, test_ratio):
    all_data.info()
    cycle = 0
    cycle_list = []

    start = 0

    for i in range(len(all_data) - 1):
        if (all_data['perc'][i+1] == 0):
            cycle += 1
            cycle_list.append(all_data.iloc[start:i])
            start = i+1

    random.shuffle(cycle_list)
    source_table = pd.concat(cycle_list, axis=0, ignore_index=True)
    source_table = source_table.drop(["lgrf", "rgrf", "l_ph_ank", "r_ph_ank","l_ph_fo","r_ph_fo","st_l"], axis = 1)
    source_table
    x = source_table
    x = x.drop(['perc'], axis=1)
    # scaler = MinMaxScaler()
    # x_scaled = scaler.fit_transform(x)
    # x_scaled = pd.DataFrame(x_scaled)

    encode_gait_percentage(source_table, 'perc')
    y = source_table[["X","Y"]]
    # data_x = x_scaled
    data_x=x.values
    x.info()
    data_y = y.values.reshape(-1,2)

    X_train, X_test,y_train, y_test = train_test_split(data_x, data_y ,
                            test_size=0.25,
                            shuffle=False)
    look_back = 10
    fore_cast = 1

    train_x, train_y = convert_data(X_train, y_train, look_back, fore_cast)
    validation_x, validation_y = convert_data(X_test, y_test, look_back, fore_cast)

    return train_x, train_y, validation_x, validation_y


import random
def get_data_frames_from_files(path, file_names, subject_dict, subjects):
    file_list=[]
    for i in range (len(file_names)):

        subject = file_names[i].split('_')[0]
        if not subject in subjects:
            continue
        else:
            print(file_names[i])
        leg_len = subject_dict[subject][0]
        weight = subject_dict[subject][1]
        tmp=pd.read_excel(path+ file_names[i], sheet_name='Sheet1')
        perc_column = tmp['perc']
        tmp = tmp.drop(columns=['perc'])
        scaler = MinMaxScaler()

    # Normalize each column separately
        normalized_data = scaler.fit_transform(tmp)
        column_names = tmp.columns

        normalized_df = pd.DataFrame(normalized_data, columns=column_names)
        tmp['l_ph_hip']=tmp['l_ph_hip']/300
        tmp['r_ph_hip']=tmp['r_ph_hip']/300
        tmp['l_ph_fo']=tmp['l_ph_fo']/300
        tmp['r_ph_fo']=tmp['r_ph_fo']/300
        tmp['lcop']= tmp['lcop']*1000
        tmp['rcop']=tmp['rcop']*1000
        tmp['strike_frame']=tmp['strike_frame']/400
        tmp['st_sw_phase']=tmp['st_sw_phase']/200
        normalized_df =tmp
        normalized_df['leg_len']=leg_len
        normalized_df['weight']=weight
        normalized_df['perc']= perc_column
        # normalized_df.insert(tmp.columns.get_loc('col1'), 'perc', perc_column)

        file_list.append(normalized_df)

    random.shuffle(file_list)

    all_data = pd.concat(file_list, axis=0, ignore_index=True)
    return all_data


from keras import Model
from keras.layers import Layer
import keras.backend as K
import keras
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                               initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        print(x.shape)
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
    
    
class Sampling(L.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var, p = inputs
        p1=tf.reduce_sum(p[0])
        p2=tf.reduce_sum(p[1])
        # print(p1)
        # print(p2)
        batch = tf.shape(z_mean)[0]
        seq = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.random.normal(shape=(batch, seq, dim))
        return z_mean*p1 + tf.exp(0.5 * z_log_var*p2) * epsilon
   
def motion_encoder_model():
    seq_len =10
    n_features=train_x.shape[2]
    inp=Input(shape=(seq_len, n_features))

def encoder_model():
    seq_len=10
    n_features=train_x.shape[2]
    x=Input(shape=(seq_len, n_features))
    part_1 = x[:, :, :n_features-2]

    part_2 = x[0, 0, n_features-2:]

    part_3 = x[:, :, n_features-5:n_features-2]
    # l2=tf.keras.layers.AveragePooling1D(
    #     pool_size=2,
    #     strides=1, padding="same")(part_3)
    lin_l2=L.Dense(8)(part_3)
    lin_l3=L.Dense(4)(lin_l2)

    # LSTM_layer2 = LSTM(16, return_sequences=True)(lin_l2)

    print(part_2)
    l1=tf.keras.layers.AveragePooling1D(
        pool_size=2,
        strides=1, padding="same")(part_1)
    lin_l1=L.Dense(4)(l1)

    att1 = attention()(part_1)
    # tmp_inp=L.Concatenate()([att1, part_2])

    rep_layer = L.RepeatVector((seq_len))(att1);
    # tmp_inp=L.Concatenate()([att1, part_2])

    # latent_sp=L.TimeDistributed(Dense(4))(rep_layer)
    # f1=tf.keras.layers.Flatten()(l1)
    # print(l1.shape)
    # lstm_inp=L.Concatenate(axis=2)([lin_l1, rep_layer, part_2])
    # RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    LSTM_layer1 = LSTM(32, return_sequences=True)(rep_layer)
    # LSTM_layer2 = LSTM(8, return_sequences=True)(LSTM_layer1)

    concat_layer=L.Concatenate(axis=2)([LSTM_layer1, lin_l3])

    # attn_layer1 = attention()(LSTM_layer2)
    mean = L.Dense(3)(concat_layer)
    log_var= L.Dense(3)(concat_layer)
    z = Sampling()([mean, log_var, part_2])
    latent_sp=L.TimeDistributed(L.Dense(4))(concat_layer)

    encoder = tf.keras.Model(x, (mean, log_var, z, latent_sp), name="Encoder")
    return encoder      

def decoder_model():
        
    latent_dim =(10,8)
    n_features=train_x.shape[2]

    n_real_features = n_features -2
    input_1_shape=(10,3)
    input_2_shape=( 10,4)
    # input_3_shape=( 10,3)
    input1 = tf.keras.Input(shape=input_1_shape, name='input_layer1')
    input2 = tf.keras.Input(shape=input_2_shape, name='input_layer2')
    concat1= L.Concatenate(axis=2)
    concatenated_input = L.Concatenate(axis=2)([input1, input2])
    # pooled_l = L.AveragePooling2D(pool_size=)
    # f=Flatten()(concatenated_input)
    # print(f)
    dec_l1 = L.Dense(8)(concatenated_input)
    # rep_layer = L.RepeatVector((seq_len))(dec_l1);
    dec_LSTM_layer1 = LSTM(32, return_sequences=True)(dec_l1)
    dec_LSTM_layer2 = LSTM(8, return_sequences=True)(dec_LSTM_layer1)

    lin_layer = L.TimeDistributed(L.Dense(n_real_features))(dec_LSTM_layer2)

    # tmp_layer = Flatten()(dec_LSTM_layer)
    # lin_layer = L.Dense(2)(tmp_layer)
    decoder = tf.keras.Model([input1, input2], lin_layer, name="Decoder")
    return decoder


seq_len=10
n_features=11
train_x=np.zeros((10,10,11))
encoder_model().summary()
decoder_model().summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, ls = self.encoder(data)
            # inp_dec = tf.concat((z, ls, p2), axis=2)
            # print(inp_dec)
            reconstruction = self.decoder([z,ls])
            # tmp1=keras.losses.binary_crossentropy(data, reconstruction)
            # reconstruction_loss = K.mean(K.square(data - reconstruction))
            mse = tf.keras.losses.MeanSquaredError()
            n_features = data.shape[2] 
            real_data = data[:,:,0:n_features-2]
            reconstruction_loss = mse(real_data, reconstruction)
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.binary_crossentropy(data, reconstruction)
            #     )
            # )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(1,2)))
            total_loss = 1.5*reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
def train_vae(train_x):
    enc=encoder_model()
    dec=decoder_model()
    vae = VAE(enc, dec)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(train_x, epochs=20, batch_size=128, verbose=2)

    return vae.encoder,vae.decoder


def test_model_get_results(encoder, mlp_model, validation_x, validation_y, display_flag, tag, file):
    print("validation_x shape",validation_x.shape)
    _,_,samp_v, ls_v=encoder.predict(validation_x)
    ls_v = tf.concat((samp_v, ls_v), axis=2)
    print('Encoded time-series shape', ls_v.shape)
    testPredict = mlp_model.predict(ls_v)
    print(validation_y.shape)
    print(testPredict.shape)
    testScore = np.sqrt(mean_squared_error(validation_y, testPredict))
    # print(testScore)
    # print('Test Score: %.2f RMSE' % (testScore))
    pred = np.zeros((len(testPredict),1))

    for iter in range(len(testPredict)):
        x = testPredict[iter][0]
        y = testPredict[iter][1]
        pred[iter] = ((math.atan2(y,x) + 2*math.pi) % (2*math.pi)) * (100 / (2*math.pi))

    actual = np.zeros((len(validation_y),1))

    for iter in range(len(validation_y)):
        x =validation_y[iter][0]
        y =validation_y[iter][1]
        actual[iter] = ((math.atan2(y,x) + 2*math.pi) % (2*math.pi)) * (100 / (2*math.pi))
    cor_actual=[]
    cor_pred=[]
    prec_list=[]
    for i in range(5):
        correct = 0
        for iter in range(len(actual)):
            if (actual[iter]>98) or(actual[iter]<3):
                correct+=1
                continue
            if (abs(actual[iter] - pred[iter]) <= (i+1)):
                correct+=1
            cor_pred.append(pred[iter])
            cor_actual.append(actual[iter])
        prec=correct * 100/len(actual)
        print("Precision ", i+1, ": ", prec)
        file.write(str(prec))
        file.write("\n")
        prec_list.append(prec)
    rmse = 0
    length = len(actual)
    for i in range(len(actual)):
        if abs (pred[i] - actual[i]) >= 90:
            length -= 1
        else:
            rmse = rmse + pow(pred[i] - actual[i], 2)
    rmse = rmse / length
    rmse = math.sqrt(rmse)
    print(rmse)
    file.write("rmse "+str(rmse))
    file.write("\n")
    if display_flag:
        plt.scatter(cor_actual, cor_pred, facecolors='none', edgecolors='crimson',alpha=0.4)
        p1 = max(max(cor_pred), max(cor_actual))
        p2 = min(min(cor_pred), min(cor_actual))

        ci = 0.1 * np.std([p1,p2]) / np.mean([p1,p2])

        plt.plot([p1, p2], [p1, p2], 'b-', linewidth =3)
        plt.title('Actual vs Prediction')
        plt.savefig(result_path+tag+"res.png")

    return prec_list, rmse


def train_mlp_model(samp_t, train_y):
    mlp_model = Sequential()

    mlp_model.add(tf.keras.Input(shape=(samp_t.shape[1], samp_t.shape[2]), name='input_layer'))
    mlp_model.add(LSTM(32))

    mlp_model.add(L.Dense(32, kernel_initializer='glorot_normal', activation='relu'))
    # mlp_model.add(L.Dense(32, kernel_initializer='glorot_normal', activation='relu', input_dim=(samp_t.shape[1]*samp_t.shape[2])))
    # mlp_model.add(L.Dense(32, kernel_initializer='glorot_normal', activation='relu', input_dim=(train_encoded.shape[1])))

    mlp_model.add(L.Dense(8, kernel_initializer='glorot_normal', activation='relu'))
    mlp_model.add(L.Dense(2))
    mlp_model.summary()

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    adam = optimizers.Adam(lr_schedule)

    mlp_model.compile(loss='mse', optimizer=adam)

    epochs = 25
    batch=64                                                                                                                                                           
    # lrate = LearningRateScheduler(step_decay)
    monitor = EarlyStopping(monitor='loss', min_delta=1e-5, patience=3)
    # train_encoded_reshaped=np.reshape(train_encoded,(train_encoded.shape[0], train_encoded.shape[1]*train_encoded.shape[2]))

    # train_encoded_reshaped=np.reshape(train_encoded,(train_encoded.shape[0], train_encoded.shape[1]))
    # train_encoded_reshaped = train_encoded
    callback_list = [monitor]
    mlp_history = mlp_model.fit(samp_t , train_y, callbacks=callback_list, epochs=epochs, batch_size=batch,  verbose=2)

    return mlp_model

def create_latent_space_train_mlp(encoder, train_x):
  print("OKAY till here 2")
  _,_,samp_t, ls_t=encoder.predict(train_x)
  print("OKAY till here 3")

  ls_t = tf.concat((samp_t, ls_t), axis=2)
  print('Encoded time-series shape', ls_t.shape)
  return ls_t

file_names = ['JL_I_0_new_.xlsx', 'JL_I_2_new_.xlsx','JL_I_3_new_.xlsx','JL_I_5_new_.xlsx','JL_I_4_new_.xlsx',
              'JS_I_1_new_.xlsx', 'JS_I_2_new_.xlsx','JS_I_3_new_.xlsx','JS_I_5_new_.xlsx','JS_I_4_new_.xlsx',
              'AK_I_0_new_.xlsx', 'AK_I_2_new_.xlsx','AK_I_3_new_.xlsx','AK_I_5_new_.xlsx','AK_I_4_new_.xlsx',
              'VN_I_0_new_.xlsx', 'VN_I_2_new_.xlsx','VN_I_3_new_.xlsx','VN_I_5_new_.xlsx','VN_I_4_new_.xlsx',
              'VP_I_0_new_.xlsx', 'VP_I_2_new_.xlsx','VP_I_3_new_.xlsx','VP_I_5_new_.xlsx','VP_I_4_new_.xlsx',
             'SOE_I_0_new_.xlsx', 'SOE_I_2_new_.xlsx','SOE_I_3_new_.xlsx','SOE_I_5_new_.xlsx', 'SOE_I_4_new_.xlsx', 'SD_I_3_new_.xlsx', 'SD_I_4_new_.xlsx','SD_I_5_new_.xlsx',
             'SD_I_1_new_.xlsx','SD_I_2_new_.xlsx','TH_I_0_new_.xlsx', 'TH_I_2_new_.xlsx', 'TH_I_3_new_.xlsx','TH_I_4_new_.xlsx', 'TH_I_5_new_.xlsx'
             ,'PK_I_0_new_.xlsx', 'PK_I_2_new_.xlsx', 'PK_I_3_new_.xlsx','PK_I_5_new_.xlsx',
              'SKS_0_I_new_.xlsx', 'SKS_2_I_new_.xlsx','SKS_3_I_new_.xlsx','SKS_4_I_new_.xlsx','SKS_5_I_new_.xlsx',
            'PH_I_0_new_.xlsx',  'PH_I_2_new_.xlsx',  'PH_I_3_new_.xlsx',  'PH_I_4_new_.xlsx',  'PH_I_5_new_.xlsx'
              ]
subject_dict = {'VN':[0.90,0.63],'AK':[0.80,0.57],'JS':[0.89,0.64],'JL':[0.79,0.63],'SKS':[0.83, 0.58],'VP':[0.93, 0.77],'SOE':[0.90, 0.83],
                'SD':[0.83, 0.70], 'TH':[0.66, 0.52], 'PK':[0.90, 0.88], 'PH':[0.92,0.77]}
subject_names = ['PH','SOE','AK','JL', 'SD','PK','TH','SKS','VP','JS','VN']#,'VN','AK' 'SOE'
sub_comb_list=[]
test_sub_list=[]
acc_list=[]
rmse_list=[]
test_acc_list=[]
test_rmse_list=[]


path="/home/vtp/Gait_Phase_Prediction/Subject_data/new_files/"
result_path = "/home/vtp/Gait_Phase_Prediction/Results/"

pkl_file=path+"all_sub_vae2_data_corr2_full.pkl"
# pkl_file=path+"good_sub_data.pkl"

for sub in subject_names:

  test_sub_list.append(sub)
  tmp=subject_names.copy()
  tmp.remove(sub)
  sub_comb_list.append(tmp)

df_dict={}

if os.path.exists(pkl_file):
    # File is already in pickle format, read to dict
    with open(pkl_file, 'rb') as file:
        df_dict = pickle.load(file)
    
else:
    # File is not in pickle format/ does not exist, convert and save it as a pickle file

    for file_name in file_names:
        subject = file_name.split('_')[0]
        if subject not in subject_names:
            continue
        leg_len = subject_dict[subject][0]
        weight = subject_dict[subject][1]
        print("Reading file :", file_name)
        tmp=pd.read_excel(path+ file_name, sheet_name='Sheet1')
        perc_column = tmp['perc']
        st_sw_col = tmp['st_sw_phase']
        sf_col = tmp['strike_frame']
        lhip_col = tmp['lhip_ang']
        rhip_col = tmp['rhip_ang']
        st_l_col = tmp['st_l']

        tmp = tmp.drop(columns=['perc', 'st_sw_phase', 'strike_frame', 'lhip_ang', 'rhip_ang', 'st_l'])
        tmp['lhip_ang'] = lhip_col
        tmp['rhip_ang'] = rhip_col
        scaler = MinMaxScaler()
        tmp['lhip_ang_n'] = scaler.fit_transform(tmp[['lhip_ang']])
        tmp['rhip_ang_n'] = scaler.fit_transform( tmp[['rhip_ang']])

        tmp['st_sw_phase'] = st_sw_col
        tmp['strike_frame'] = sf_col
        tmp['st_l'] = st_l_col

        column_names = tmp.columns

        tmp['l_ph_hip']=tmp['l_ph_hip']/300
        tmp['r_ph_hip']=tmp['r_ph_hip']/300
        tmp['lhip_ang']=tmp['lhip_ang']/300
        tmp['rhip_ang']=tmp['rhip_ang']/300

        # tmp['l_ph_fo']=tmp['l_ph_fo']/300
        # tmp['r_ph_fo']=tmp['r_ph_fo']/300
        tmp['lcop']= tmp['lcop']*1000
        tmp['rcop']=tmp['rcop']*1000
        tmp['strike_frame']=tmp['strike_frame']/300
        tmp['st_sw_phase']=tmp['st_sw_phase']/200
        normalized_df =tmp

        normalized_df['leg_len']=leg_len
        normalized_df['weight']=weight
        normalized_df['perc']= perc_column

        df_dict[subject] = normalized_df
    with open(pkl_file,'wb') as pickle_file:
        pickle.dump(df_dict, pickle_file)

with open(result_path+"all_results_vae2.txt","w") as file:  
    for sub_iter,sub_comb in enumerate(sub_comb_list):
        print("Subject combination :",sub_comb)
        sub_tag=test_sub_list[sub_iter]
        # all_data=get_data_frames_from_files(path, file_names, subject_dict, sub_comb)
        file_list=[]
        for sub in sub_comb:
          tmp_df=df_dict[sub]
          file_list.append(tmp_df)
        random.shuffle(file_list)
        all_data = pd.concat(file_list, axis=0, ignore_index=True)
        train_x, train_y, validation_x, validation_y = get_train_data_from_df(all_data, 0.25)
        print(train_x.shape)
        print(validation_x.shape)
        print("SSS")
        seq_len = train_x.shape[1]
        n_features = train_x.shape[2]
        encoder=[]
        decoder=[]
        mlp_model=[]
        encoder,decoder = train_vae(train_x)
        print("OKAY till here 1")

        samp_t=create_latent_space_train_mlp(encoder, train_x)
        mlp_model = train_mlp_model(samp_t, train_y)
        file.write("Training Result :")
        file.write("\n")
        acc, rmse=test_model_get_results(encoder,mlp_model, validation_x, validation_y, False, sub_tag, file)
        acc_list.append(acc)
        rmse_list.append(rmse)
        print("testing on :", test_sub_list[sub_iter])
        file.write("Testing  Result :")
        file.write(test_sub_list[sub_iter])
        file.write("\n")
        # all_data=get_data_frames_from_files(path, file_names, subject_dict, [test_sub_list[sub_iter]])
        all_data = df_dict[test_sub_list[sub_iter]]
        test_x, test_y, validation_x, validation_y = get_train_data_from_df(all_data, 0.25)
        acc, rmse=test_model_get_results(encoder,mlp_model, test_x, test_y, True, sub_tag, file)
        test_acc_list.append(acc)
        test_rmse_list.append(rmse)
        # break
        # file.write(acc)
        file.write("\n")
        # if sub_iter>=5:
        break
    file.close()
