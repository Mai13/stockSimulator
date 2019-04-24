import trade
import strategies


def main ():

    algorithm_data, test_data = preprocessing.ReadData()

    wallet = trade.Porfolio(initial_money=1000)

    strategy_one = strategies.ml()
    strategy_two = strategies.dl()

    for row in test_data.shape[0]

        strategy_one.predict()
        strategy_two.predict()


if __name__ == "__main__":

    main()