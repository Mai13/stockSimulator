class ml():

    def __init__(self):

        self.model = 0

    def train(self, X_train, y_train):


        self.model = self.model.fit(X_train, y_train)

    def grid_search(self, X_train, y_train, fold_number=5,
                    overfitting_threshold=0.1,*parameters):

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
                                                  criterion='entropy')
                    model.fit(x_splited_train, y_splited_train)
                    predicted_val = model.predict(x_val)
                    predicted_train = model.predict(x_splited_train)
                    rmse_train =+ math.sqrt(mean_squared_error(y_splited_train, predicted_train))
                    rmse_test =+ math.sqrt(mean_squared_error(y_val, predicted_val))

                if float(rmse_test / fold_number) > best_rsme and abs(rmse_test/fold_number - rmse_train/fold_number) < overfitting_threshold:

                    best_rsme = rmse_test / fold_number
                    best_parameter_1 = parameter_1
                    best_parameter_2 = parameter_2

        self.model = RandomForestRegressor(random_state=2018,
                                            n_estimators=best_parameter_1,
                                            max_features='auto',
                                            max_depth=best_parameter_2,
                                            criterion='entropy')


    def predict (self, ticker, x_to_predict):

        return {ticker: float(self.model.predict(x_to_predict))}



class dl():

    def __init__(self):


        self.model = None
        self.parameters = None

    def grid_search(self, data, fold_number=5, **parameters):

        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=fold_number)

        # self.model =

    def train(self, data, parameters):

        if self.parameters == None:

            print('There is not an optimum model, you need to tri grid search first')

        # else:





        #TODO: train the algorithm and append the results to logger


        # pass

    def predict (self):

        prices = 10

        return prices

