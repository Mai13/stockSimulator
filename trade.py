class Porfolio():

    def __init__(self, initial_money):

        self.money_to_spend = initial_money
        self.data_path = 'some url path'
        self.current_actions = []

    def optimize (self):

        if self.current_actions:
            for action, price in self.current_actions:

                print(action, price)
        else:

            print('no current actions %s'%(len(self.current_actions)))