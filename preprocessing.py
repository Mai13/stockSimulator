import pandas as pd


class ReadData:

    def __init__(self):

        self.raw_data_path = '/Users/maialenberrondo/Documents/stockSimulator/data/'
        self.train_datasets = []
        self.test_datasets = []

    def load_data(self, tickers):

        for ticker in tickers:
            self.train_datasets.append(pd.read_csv(self.raw_data_path + str(ticker+'train_val_nan_droped.csv')))
            self.test_datasets.append(pd.read_csv(self.raw_data_path + str(ticker+'test_val_nan_droped.csv')))

    def preprocess(self, tickers):

        from sklearn import preprocessing

        self.load_data(tickers)
        train_data = {}
        cnt = 0

        for train_dataframe in self.train_datasets:

            train_dataframe['date'] = pd.to_datetime(train_dataframe['date']).astype(int) / 10**9
            train_dataframe.drop(['Unnamed: 0', '1. open', '2. high', '3. low', 'hour', 'time', 'day_of_week'], axis=1,
                                 inplace=True)
            train_dataframe.rename(columns={'4. close': 'price', '5. volume': 'volume'}, inplace=True)
            train_dataframe['target'] = train_dataframe['price'].shift(-1, axis=0)
            train_dataframe.dropna(inplace=True)
            for col in train_dataframe.columns:
                if col != 'target' and col != 'date':
                    train_dataframe[col] = preprocessing.scale(train_dataframe[col].values)

            train_data[tickers[cnt]] = train_dataframe
            cnt += 1

        test_data = {}
        cnt = 0

        for test_dataframe in self.test_datasets:

            test_dataframe['date'] = pd.to_datetime(test_dataframe['date']).astype(int) / 10 ** 9
            test_dataframe.drop(['Unnamed: 0', '1. open', '2. high', '3. low', 'hour', 'time', 'day_of_week'], axis=1,
                                 inplace=True)
            test_dataframe.rename(columns={'4. close': 'price', '5. volume': 'volume'}, inplace=True)
            test_dataframe['target'] = test_dataframe['price'].shift(-1, axis=0)
            test_dataframe.dropna(inplace=True)
            for col in test_dataframe.columns:
                if col != 'target' and col != 'date':
                    test_dataframe[col] = preprocessing.scale(test_dataframe[col].values)
            test_data[tickers[cnt]] = test_dataframe
            cnt += 1

        return train_data, test_data
