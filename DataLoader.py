import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random


class DataLoader():

    def __init__(self, path):
        self.train_path=path
        self.look_back=10

    def encode_gait_percentage(self, df, name):
        percent = df[name].values.tolist()
        X = np.zeros((len(df[name]),1))
        Y = np.zeros((len(df[name]),1))
        for i in range (len(percent)): 
            phi = percent[i] * 2 * math.pi / 100
            X[i] = math.cos(phi)
            Y[i] = math.sin(phi)
        df['X'] = X
        df['Y'] = Y

    def prepare_data(self, train_file_names, test_file_names):
        t_data=[]
        t_target=[]
        for f_name in train_file_names:
            name=self.path+f_name
            source_table = pd.read_excel(name, sheet_name='Sheet1')
            source_table.dropna()
            source_table.keys()
            source_table  
            x = source_table
            # x = x.drop(['perc'], axis=1)
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x)
            x_scaled = pd.DataFrame(x_scaled)

            self.encode_gait_percentage(source_table, "perc")
            y = source_table[["X","Y"]] 
            source_table.info()
            source_table.describe().T

            data_x = source_table.drop(["perc", "X", "Y","n_lgrf","n_rgrf"], axis = 1)
            target = source_table[["X", "Y"]]
            # target = source_table[["perc"]]

            data_x.info()
            target.info()

            scaler = MinMaxScaler()
            data = scaler.fit_transform(data_x)
            # ratio = 0.8
            # training_cutoff = math.floor(ratio * len(source_table))

            t_data.append(data)
            t_target.append(target.values)
            print("One file DOneeee!!!!")

        training_data=np.concatenate((t_data[0], t_data[1]), axis=0) 
        training_target=np.concatenate((t_target[0], t_target[1]), axis=0)
        # training_data=data
        # training_target=target.values

        name=self.path +test_file_names

        source_table = pd.read_excel(name, sheet_name='Sheet1')
        source_table.dropna()
        source_table.keys()
        source_table.info()  
        x = source_table
        # x = x.drop(['perc'], axis=1)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        x_scaled = pd.DataFrame(x_scaled)

        self.encode_gait_percentage(source_table, "perc")
        y = source_table[["X","Y"]] 
        source_table.info()
        source_table.describe().T

        data_x = source_table.drop(["perc", "X", "Y","lgrf","r_grf"], axis = 1)
        target = source_table[["X","Y"]]
        print("heklkkioi")
        data_x.info()
        target.info()

        scaler = MinMaxScaler()
        data = scaler.fit_transform(data_x)
        validation_data = data
        validation_target = target.values


        print(training_data.shape)
        print(training_target.shape)
        print(validation_data.shape)
        print(validation_target.shape)
        return training_data, training_target, validation_data, validation_target


    def convert_data(self, d_x, d_y, look_back = 1, forecast = 1):
        dataX = []
        dataY = []

        for i in range(look_back, len(d_x) - forecast):
            dataX.append(d_x[i - look_back: i]) 
            dataY.append(d_y[i + forecast,:])
        return np.array(dataX), np.array(dataY)


    def GetTrainTestData(self, path="/home/vtp/Gait_Phase_Prediction/Subject_data/best_xls", multi_dat=True):

        self.path=path
        file_name=['MS_1.xlsx', 'MS_2.xlsx']
        test_file='MS_3.xlsx'
        if not multi_dat:
            training_data, training_target, validation_data, validation_target =self.prepare_data( file_name, test_file)
            look_back = 10
            fore_cast = 1

            self.train_x, self.train_y = self.convert_data(training_data, training_target, self.look_back)
            self.validation_x, self.validation_y = self.convert_data(validation_data, validation_target, look_back, fore_cast)
            print("X data",self.train_x.shape)
            print("Y data",self.train_y.shape)
        else:
            path='/home/vtp/Gait_Phase_Prediction/Subject_data/best_xls/'
            path='/home/vtp/Gait_Phase_Prediction/Subject_data/inclined_data/'

            self.path=path
            # file_name=['MS_1.xlsx', 'SOE_1.xlsx','MS_1.xlsx','VP_3.xlsx' ,'SOE_1.xlsx','MS_1.xlsx','SOE_2.xlsx', 'PH_SPT3.xlsx','PH_SPT4.xlsx','VP_1.xlsx','VP_2.xlsx','PH_SPT2.xlsx', 'SOE_3.xlsx']
            file_name=['SD_1_I.xlsx', 'SD_3_I.xlsx', 'SD_4_I.xlsx', 'SD_5_I.xlsx', 'SD_2_I.xlsx','SD_2.xlsx','SD_1.xlsx', 'SD_3.xlsx', 'MS_1.xlsx', 'SOE_1.xlsx','MS_1.xlsx','VP_3.xlsx' ,'SOE_1.xlsx','MS_1.xlsx','SOE_2.xlsx', 'PH_SPT3.xlsx','PH_SPT4.xlsx','VP_1.xlsx','VP_2.xlsx','PH_SPT2.xlsx', 'SOE_3.xlsx']
            test_file='SOE_1fin3.xlsx'
            data_x,data_y= self.shuffle_multiple_datasets_based_on_gait_cycle( file_name, path, test_file)
            X_train, X_test,y_train, y_test = train_test_split(data_x, data_y ,
                                   test_size=0.25, 
                                   shuffle=False)
            look_back = 10
            fore_cast = 1

            self.train_x, self.train_y = self.convert_data(X_train, y_train, look_back, fore_cast)
            self.validation_x, self.validation_y = self.convert_data(X_test, y_test, look_back, fore_cast)
            print("X data",self.train_x.shape)
            print("Y data",self.train_y.shape)
            print("Validation X data",self.validation_x.shape)
            print("Validation Y data",self.validation_y.shape)
            

    def shuffle_multiple_datasets_based_on_gait_cycle(self, file_names, path_name, test_file_name):
        t_data=[]
        t_target=[]
        t_perc=[]

        test_data=[]
        test_target=[]

        file_iter=0
        
        for f_name in file_names:
        
            name=path_name+f_name
            print(name)
            source_table = pd.read_excel(name, sheet_name='Sheet1')

            source_table.keys()
            source_table
            x = source_table
            # x = x.drop(['perc_new'], axis=1)
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x)
            x_scaled = pd.DataFrame(x_scaled)

            self.encode_gait_percentage(source_table, "perc_new")
            y = source_table[["X","Y"]] 
            source_table.info()
            source_table.describe().T

            data_x = source_table.drop(["perc_new", "X", "Y","n_lgrf","n_rgrf","l_ph_ank","r_ph_ank"], axis = 1)
            target = source_table[["X", "Y"]]
            target_perc=source_table[["perc_new"]]
            data_x.info()
            target.info()

            scaler = MinMaxScaler()
            data = scaler.fit_transform(data_x)
            t_data.append(data)
            t_target.append(target.values)
            t_perc.append(target_perc.values)

            # ratio = 0.8
            # training_cutoff = math.floor(ratio * len(source_table))

            
            print("One file DOneeee!!!!")
            

        #create list of numpy based on gait cycle and shuffle x and y together


        iter=0
        random_data_x=[]
        random_data_y=[]
        main_iter=0

        for l_data in t_perc:
            start=0
            end_flag=True
            end=0
            iter=0
            l=len(l_data)

            for row_perc in l_data: 
                if row_perc[0]==0 and not end_flag:
                    start=iter
                    end_flag=True
                if row_perc[0]==100 and end_flag:      
                    end=iter+1;
                    end_flag=False;
                    random_data_y.append(t_target[main_iter][start+1:end,:])    
                    random_data_x.append(t_data[main_iter][start+1:end,:])    

                iter+=1
            main_iter+=1


        zipped = list(zip(random_data_x, random_data_y))
        random.shuffle(zipped)
        shuffled_x, shuffled_y = zip(*zipped)
        data_x = np.vstack(shuffled_x)
        data_y = np.vstack(shuffled_y)

        return data_x,data_y
path="/home/vtp/masters_proj/GaitPhase/Subject_data/"

# data=DataLoader(path)
# data.GetTrainTestData()
 
    

