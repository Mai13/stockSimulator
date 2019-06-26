
import strategies
import preprocessing
import trade


# TODO: set linter
# TODO: set logger

def main ():

    data = preprocessing.ReadData()


    algorithm_data, test_data = data.preprocess()

    wallet = trade.Porfolio(initial_money=100000)

    strategy_one = strategies.ml()  # algorithm_data
    strategy_two = strategies.dl()  # algorithm_data

    # TODO: loop to keep doing it in every minute in test data

    """
    
    wallet_along_time = []
    
    for row in range(test_data.shape[0]):

        strategy_one.predict()
        strategy_two.predict()
        wallet_along_time.append(wallet.optimize())
        strategy_one.fit(+row)
        strategy_two.fit(+row)

    """
    strategy_one.predict()
    strategy_two.predict()

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