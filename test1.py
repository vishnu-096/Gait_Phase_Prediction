from CNN_LSTM import *
from DataLoader import *

train_cnnlstm=model(4)
train_cnnlstm.pop_data(True)
# train_cnnlstm.load_dataset()
train_cnnlstm.train_model(4)
train_cnnlstm.test_model(False)

# train_cnnlstm=model(4)
# train_cnnlstm.pop_data(True)
# train_cnnlstm.test_model(True)





