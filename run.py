
import strategies
import preprocessing
import trade


# TODO: set linter
# TODO: set logger

def main ():

    data = preprocessing.ReadData()


    algorithm_data, test_data = data.preprocess()

    wallet = trade.Porfolio(initial_money=1000)

    strategy_one = strategies.ml()
    strategy_two = strategies.dl()

    # TODO: loop to keep doing it in every minute

    """
    
    for row in range(test_data.shape[0]):

        strategy_one.predict()
        strategy_two.predict()
        wallet.optimize()

    """
    strategy_one.predict()
    strategy_two.predict()

    # TODO:link the models with the optimization strategy

    predictions = [{"AAPL": [10, 0.98]}, {"xdf": [120, 0.6]}]
    # wallet.optimize()#  solve problem with self
    wallet.optimize(predictions)

if __name__ == "__main__":

    main()