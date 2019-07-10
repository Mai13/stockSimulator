class ml():

    def __init__(self):

        self.model = {}

    def train(self, X_train, y_train, ticker):
        import numpy as np

        # print(self.model)
        # print(self.model.get(ticker))
        """
        print(type(y_train))
        print(y_train)
        print(y_train[0])
        print(f'TRAIN ___ shapes X: {X_train.shape}, y: {y_train.shape}, y: {np.array(y_train).reshape(1, -1)}')
        """

        self.model[ticker] = self.model.get(ticker).fit(X_train, y_train)

    def grid_search(self, X_train, y_train, ticker, fold_number=5,
                    overfitting_threshold=0.1, parameters=[[20], [5]]):
        # print(f'the ticker of the GRID SEARCH {ticker}')
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        import math

        tscv = TimeSeriesSplit(n_splits=fold_number)

        best_parameter_1 = 0
        best_parameter_2 = 0
        best_rsme = 0

        for parameter_1 in parameters[0]:
            for parameter_2 in parameters[1]:

                rmse_train = 0
                rmse_test = 0

                for train_index, test_index in tscv.split(X_train):
                    x_splited_train, x_val = X_train[train_index], X_train[test_index]
                    y_splited_train, y_val = y_train[train_index], y_train[test_index]

                    model = RandomForestRegressor(random_state=2018,
                                                  n_estimators=parameter_1,
                                                  max_features='auto',
                                                  max_depth=parameter_2,
                                                  ) # criterion='entropy'
                    # print(f'shapes X: {x_splited_train.shape}, y: {y_splited_train.shape}')
                    model.fit(x_splited_train, y_splited_train.ravel())
                    predicted_val = model.predict(x_val)
                    predicted_train = model.predict(x_splited_train)
                    rmse_train =+ math.sqrt(mean_squared_error(y_splited_train, predicted_train))
                    rmse_test =+ math.sqrt(mean_squared_error(y_val, predicted_val))
                    # print(parameter_1, parameter_2, rmse_test, rmse_train, abs(rmse_train-rmse_test))
                """
                print(f'first condition {float(rmse_test / fold_number)},'
                    f'second condition {abs(rmse_test/fold_number - rmse_train/fold_number)},'
                      f'Train rsem {rmse_train}, Test rsme {rmse_test}')
                """
                if float(rmse_test / fold_number) > best_rsme and abs(rmse_test/fold_number - rmse_train/fold_number) < overfitting_threshold:
                    # print('hello')
                    best_rsme = rmse_test / fold_number
                    best_parameter_1 = parameter_1
                    best_parameter_2 = parameter_2

        # print(f'Best parameter 1: {best_parameter_1}, Best parameter 2: {best_parameter_2}')

        if best_rsme == 0 and best_parameter_1 == 0 and best_parameter_2 == 0:

            print(f'The error is TRAIN {rmse_train/fold_number}, The error in TEST {rmse_test/fold_number}')
            print(f'There is probabliy an overfitting error, the simplest model will be applied')
            best_parameter_1 = parameters[0][0]
            best_parameter_2 = parameters[1][0]

        self.model[ticker] = RandomForestRegressor(random_state=2018,
                                            n_estimators=best_parameter_1,
                                            max_features='auto',
                                            max_depth=best_parameter_2,
                                            ) # criterion='entropy'
        # print(ticker)
        # print(self.model)
    def predict(self, x_to_predict, ticker):

        return self.model.get(ticker).predict(x_to_predict)



class dl():

    def __init__(self):


        self.model = {}
        self.sequence_lenght = 480 * 7

    def preprocessing(self, x_data, y_data):

        x, y = []

        for i in range(int(round(self.sequence_lenght)), x_data.shape[0]):

            x.append(x_data.iloc[i-int(round(self.sequence_lenght)):i, :])
            y.append(y_data.iloc[i,:])

        return x, y

    def grid_search(self, X_train, y_train, ticker):

        sequenced_x, sequenced_y = self.preprocessing(X_train, y_train)

        import tensorflow as ts


        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1, activation='linear'))

        for learning_rate in [0.001, 0.01, 0.1]:

            opt = adam(lr=learning_rate, decay=1e-6)
            model.compile(
                loss=tf.keras.metrics.mean_squared_error,
                optimizer=opt,
                metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            )
            model.fit(
                # TODO: Implement
            )




    def train(self, data, parameters):


    def predict (self, data):



