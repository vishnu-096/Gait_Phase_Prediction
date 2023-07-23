from CNN_LSTM import *
from DataLoader import *
from sklearn.model_selection import KFold

only_inclined =True

train_cnnlstm=model(5)
train_cnnlstm.pop_data(True)
# train_cnnlstm.load_dataset()
train_cnnlstm.train_model(5)
train_cnnlstm.test_model(False)

# train_cnnlstm=model(4)s
# train_cnnlstm.pop_data(True)
# train_cnnlstm.test_model(True)


def k_fold_cross_validation(k):

    path='/home/vtp/Gait_Phase_Prediction/Subject_data/inclined_data/'

    #'SD_1_I.xlsx', 'SD_3_I.xlsx', 'SD_4_I.xlsx', 'SD_5_I.xlsx', 'SD_2_I.xlsx'
    # file_name=['MS_1.xlsx', 'SOE_1.xlsx','MS_1.xlsx','VP_3.xlsx' ,'SOE_1.xlsx','MS_1.xlsx','SOE_2.xlsx', 'PH_SPT3.xlsx','PH_SPT4.xlsx','VP_1.xlsx','VP_2.xlsx','PH_SPT2.xlsx', 'SOE_3.xlsx']
    file_name=['SD_2.xlsx','SD_1.xlsx', 'SD_3.xlsx', 'MS_1.xlsx', 'SOE_1.xlsx','MS_1.xlsx','VP_3.xlsx' ,'SOE_1.xlsx','MS_1.xlsx','SOE_2.xlsx', 'PH_SPT3.xlsx','PH_SPT4.xlsx','VP_1.xlsx','VP_2.xlsx','PH_SPT2.xlsx', 'SOE_3.xlsx']
    inclined_files = ['SD_1_I.xlsx', 'SD_3_I.xlsx', 'SD_4_I.xlsx', 'SD_5_I.xlsx', 'SD_2_I.xlsx', 'SKS_0_I.xlsx',  'SKS_2_I.xlsx',  'SKS_3_I.xlsx',  'SKS_4_I.xlsx', 'PK_3_I.xlsx', 'PK_5_2_I.xlsx', 'SKS_5_I.xlsx', 'PK_0_I.xlsx', 'PK_2_I.xlsx']
    if only_inclined:
        file_name = inclined_files
    else:
        file_name = file_name+inclined_files
    test_file='SOE_1fin3.xlsx'
    data_x,data_y= train_cnnlstm.data.shuffle_multiple_datasets_based_on_gait_cycle( file_name, path, test_file)


    # Define number of folds for cross-validation
    k = 5

    # Create KFold object
    kf = KFold(n_splits=k, shuffle=False)

    # Initialize lists to store cross-validation results
    train_scores = 0
    test_scores = 0
    accuracy = []
    iter=0
    mean_acc=[]
    with open(path+'all_inclined_data_results_3_cop_hip.txt', 'w') as f:
        for train_index, test_index in kf.split(data_x):
        # Split data into training and validation sets for this fold
            X_train_fold, y_train_fold = train_cnnlstm.data.convert_data(data_x[train_index], data_y[train_index], 10, 1)
            X_val_fold, y_val_fold = train_cnnlstm.data.convert_data(data_x[test_index], data_y[test_index], 10, 1)
            train_cnnlstm.data.train_x=X_train_fold
            train_cnnlstm.data.validation_x = X_val_fold
            train_cnnlstm.data.train_y =y_train_fold
            train_cnnlstm.data.validation_y = y_val_fold
            train_cnnlstm.train_model(3, iter)
            train,test,acc=train_cnnlstm.test_model(False, "3_all_data_inclined_cop_hip_"+str(iter))

            iter+=1

            train_scores+=train
            test_scores+=test
            # accuracy.append(acc)
            iter_acc=0
            for el in acc:
                if iter==1:
                    mean_acc.append(el)
                else:
                    mean_acc[iter_acc]+=el
                iter_acc+=1
        f.write(str(train/k)+'\n')
        f.write(str(test/k)+'\n')

        f.write('\n\n')    
        for el in mean_acc:
            f.write(str(el/k)+'  ')


 
# k_fold_cross_validation(5)
print("done!")


