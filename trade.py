class Porfolio():

    def __init__(self, initial_money):

        self.money_to_spend = initial_money
        self.data_path = 'some url path'
        self.current_actions = []

    def buy(self, ticker, quantity, actual_price, stop_rate=0.03):

        self.money_to_spend = self.money_to_spend - quantity*actual_price
        # current action structure exaple {"AAPL": [120, 6, 110]} buying price, quantity, stop
        self.current_actions = self.current_actions.append({ticker: [actual_price, quantity, actual_price*stop_rate]}) # TODO: see if we insert a stop


        pass

    def sell(self):
        pass

    def do_nothing(self):
        pass

    def optimize(self, predictions):

        if self.current_actions:
            for ticker, price in self.current_actions: # ticker the symbol that is used in the stock

                print(ticker, price)
        else:
            print('no current actions %s'%(len(self.current_actions)))
            # print(predictions)
            for dictionary in predictions:
                for ticker, value in dictionary.items():
                    # TODO: base on game theory and papers

                    print(key, value)
                # print(b[0], b[1])
