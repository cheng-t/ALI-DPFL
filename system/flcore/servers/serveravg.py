
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import mindspore as ms
import numpy as np
import sys

class FedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        
        self.set_clients(clientAVG)

    def train(self):

        self.send_models()

        for i in range(self.global_rounds+1):

            self.selected_clients = self.select_clients()
            
            print(i,':',end = '') 
            sys.stdout.flush()
            for client in self.selected_clients:
                client.train()
                # pass

            self.aggregate_parameters()

            self.send_models()

            if i % self.eval_gap == 0:  
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                acc,loss = self.evaluate()
                print(f'Global model round {i} accuracy: {acc}')
                print(f'loss {loss}')

    def test(self):
        acc,loss = self.evaluate()
        print('Training Finished')
        print(f'Final Accuracy: {acc}')
        print(f'Final Loss: {loss}')


            





