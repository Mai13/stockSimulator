import strategies
import preprocessing
import trade
import matplotlib.pyplot as plt
import sys


def main():

    data = preprocessing.ReadData()
    tickers = ['AAPL', 'HPE', 'RHT', 'STX', 'WDC']
    algorithm_data, simulation_data = data.preprocess(tickers)

    wallet_ml = trade.Porfolio(initial_money=100000)

    cnt = 0
    period_between_re_tain = 480
    money_vs_time_ml = []
    ml_strategy = strategies.ml()
    print(f'total time is {simulation_data.get("AAPL").shape[0]}')
    # for minute in range(simulation_data.get('AAPL').shape[0]):
    for minute in range(3):
        ml_predictions = {}
        current_values = {}
        for ticker in tickers:

            try:
                print(cnt)
                if cnt == 0 or float(cnt / period_between_re_tain).is_integer():
                    ml_strategy.grid_search(algorithm_data.get(ticker)[['price', 'volume']].values,
                                            algorithm_data.get(ticker)[['target']].values,
                                            ticker,
                                            5,
                                            0.1,
                                            [[10, 30, 50, 70, 100], [3, 5, 7, 10, 15, 20]])
                    ml_strategy.train(algorithm_data.get(ticker)[['price', 'volume']].values,
                                      algorithm_data.get(ticker)[['target']].values,
                                      ticker)
                    ml_predictions[ticker] = ml_strategy.predict(simulation_data.get(ticker).iloc[0, 1:3].values.reshape(1, -1),
                                                                 ticker)
                else:

                    ml_strategy.train(algorithm_data.get(ticker)[['price', 'volume']].values,
                                      algorithm_data.get(ticker)[['target']].values,
                                      ticker)
                    ml_predictions[ticker] = ml_strategy.predict(
                        simulation_data.get(ticker).iloc[0, 1:3].values.reshape(1, -1), ticker)

                current_values[ticker] = algorithm_data.get(ticker).iloc[-1, 3]
                algorithm_data[ticker] = algorithm_data.get(ticker).append(simulation_data.get(ticker).iloc[0, :], ignore_index=True)
                simulation_data.get(ticker).drop(simulation_data.get(ticker).head(1).index, inplace=True)
            except:
                print(f'the simulation stopped in line {cnt}')
                print(f'Unexpected error:{sys.exc_info()[0]}')
                break
        cnt += 1
        money_ml = wallet_ml.optimize(ml_predictions, current_values, tickers)
        money_vs_time_ml.append(money_ml)
    print(current_values)
    money_vs_time_ml.append(wallet_ml.close_all(current_values, tickers))

    plt.figure(figsize=(20, 10))
    plt.plot(range(0, 4), money_vs_time_ml) # simulation_data.get('AAPL').shape[0] + 1
    plt.savefig('time_evolution.png')

if __name__ == "__main__":

    main()