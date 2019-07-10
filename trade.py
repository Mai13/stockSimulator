class Porfolio():

    def __init__(self, initial_money):

        self.money_to_spend = initial_money
        self.open_positions = {}
        self.current_actions = {}

    def buy(self, ticker, quantity, current_value, stop_rate=0.03):

        self.money_to_spend = self.money_to_spend - quantity*current_value
        # current action structure exaple {"AAPL": [120, 6, 110]} buying price, quantity, stop
        print([current_value, quantity, current_value*stop_rate])
        self.open_positions[ticker] = [current_value, quantity, current_value*stop_rate]


    def sell(self, ticker, quantity, current_value):

        self.money_to_spend = self.money_to_spend + quantity*current_value
        del self.current_actions[ticker]

    def check_investments(self, predictions, ticker, ticker_info, current_values):

        predicted_value = predictions.get(ticker)
        current_value = current_values.get(ticker)

        if predicted_value < ticker_info[2]:  # If stop will be jumped
            self.sell(ticker, ticker_info[1], current_value)
        elif predicted_value < ticker_info[0]:  # Negative value
            self.sell(ticker, ticker_info[1], current_value)

    def optimize(self, predictions, current_values, tickers):

        for ticker in tickers:

            if ticker in self.current_actions.keys():

                self.check_investments(predictions, ticker, self.current_actions.get(ticker), current_values)
            else:

                if predictions.get(ticker) > current_values.get(ticker):
                    predicted_earnings = (predictions.get(ticker)-current_values.get(ticker)) / current_values.get(ticker)
                    if predicted_earnings < 0.5:
                        """
                        print('heree')
                        print(self.money_to_spend * 0.5 * predicted_earnings)
                        print('current value is', current_values.get(ticker))
                        print((self.money_to_spend * 0.5 * predicted_earnings) / float(current_values.get(ticker)))
                        """
                        number_of_actions = int(self.money_to_spend * 0.5 * predicted_earnings / float(current_values.get(ticker)))
                    else:
                        number_of_actions = int(self.money_to_spend * 0.5 / current_values.get(ticker)) # In case earnings of 500% are predicted
                    # print(ticker, number_of_actions, current_values.get(ticker))
                    self.buy(ticker, number_of_actions, current_values.get(ticker))

        return self.money_to_spend

    def close_all(self, current_values, tickers):

        for ticker in tickers:

            if ticker in self.current_actions.keys():

                self.sell(ticker, self.current_actions.get(ticker), current_values.get(ticker))

        return self.money_to_spend





