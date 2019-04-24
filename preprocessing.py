import pandas as pd

class ReadData():

    def __init__(self):
        #TODO: set path of configuration file
        self.raw_data_path = '/Users/maialenberrondo/Documents/stockSimulator/data/'
    def preprocess(self):
        #TODO: create all the preprocessing
        train_data = pd.read_csv(self.raw_data_path + str('AAPLtrain_val_nan_droped.csv'))
        test_data = pd.read_csv(self.raw_data_path + str('AAPLtest_val_nan_droped.csv'))
        return train_data, test_data