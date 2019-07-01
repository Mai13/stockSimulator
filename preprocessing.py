import pandas as pd

class ReadData():

    def __init__(self):

        self.raw_data_path = '/Users/maialenberrondo/Documents/stockSimulator/data/'
        self.train_datasets = []
        self.test_datasets = []

    def load_data(self, tickers):

        # TODO: Create the time series unix time as index, current value as input ans curent value shuffled as target

        for ticker in tickers:
            self.train_datasets.append(pd.read_csv(self.raw_data_path + str(ticker+'train_val_nan_droped.csv')))
            self.test_datasets.append(pd.read_csv(self.raw_data_path + str(ticker+'train_val_nan_droped.csv')))



    def preprocess(self, tickers):

        for path_train_dataframe in self.train_datasets:
            df = pd.read_csv(path_train_dataframe)
            print(df.columns)
        #TODO: create all the preprocessing
        train_data = pd.read_csv(self.raw_data_path + str('AAPLtrain_val_nan_droped.csv'))
        test_data = pd.read_csv(self.raw_data_path + str('AAPLtest_val_nan_droped.csv'))
        return train_data, test_data