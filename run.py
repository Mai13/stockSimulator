import strategies
import preprocessing
import trade


def main ():

    data = preprocessing.ReadData()
    tickers = ['AAPL', 'HPE', 'RHT', 'STX', 'WDC']
    algorithm_data, simulation_data = data.preprocess(tickers)

    wallet = trade.Porfolio(initial_money=100000)

    # print(algorithm_data.head())
    # print(algorithm_data.columns)

    ml_predictions = {}
    cnt = 0
    # TODO: The Time loops
    print(f'total time is {simulation_data.get("AAPL").shape[0]}')
    for minute in range(simulation_data.get('AAPL').shape[0]):
    for ticker in tickers:
        if cnt == 0:
            print(f'-------   {ticker}   -------')
            ml_strategy = strategies.ml()
            ml_strategy.grid_search(algorithm_data.get(ticker)[['price', 'volume']].values,
                                    algorithm_data.get(ticker)[['target']].values,
                                    5,
                                    0.1,
                                    [[10, 30, 50, 70, 100], [3, 5, 7, 10, 15, 20]])
            ml_strategy.train(algorithm_data.get(ticker)[['price', 'volume']].values,
                              algorithm_data.get(ticker)[['target']].values)
            ml_predictions[ticker] = ml_strategy.predict(simulation_data.get(ticker)[['price', 'volume']].values.reshape(1, -1))[0]
            # TODO: remove first row from simulation
            # TODO: Add that data to algorithm data
            # TODO: Current value (price)
            cnt += 1
        elif cnt == 480:
            cnt = 1
        else:
            # TODO: ONLY TRAIN
        # TODO: INCLUDE PERFORM STRATEGY
    print(ml_predictions)

    # strategy_one = strategies.ml()  # algorithm_data
    # strategy_two = strategies.dl(algorithm_data)  # algorithm_data

    # TODO: loop to keep doing it in every minute in test data
    # TODO: start the grid search every "night"
    # TODO: Preprocessing has no shuffle if LSTM needs it we will do it later
    """
    
    wallet_along_time = []
    
    for row in range(test_data.shape[0]):

        strategy_one.predict()
        strategy_two.predict()
        wallet_along_time.append(wallet.optimize())
        strategy_one.fit(+row)
        strategy_two.fit(+row)

    """
    # strategy_one.predict()
    # strategy_two.predict()

    # TODO:link the models with the optimization strategy

    predictions = {"AAPL": 10, "xdf": 120}
    current_values = {"AAPL": 15, "xdf": 110}
    tickers = ["AAPL", "xdf"]
    # wallet.optimize()#  solve problem with self
    for cnt in range(10):
        money = wallet.optimize(predictions, current_values, tickers)
        print('money in iteration %s: %s' % (cnt, money))

    total_amount = wallet.close_all(current_values, tickers)
    print('total amount of money: %s' % total_amount)

if __name__ == "__main__":

    main()