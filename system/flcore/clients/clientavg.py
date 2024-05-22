from flcore.clients.clientbase import Client

import numpy as np
import mindspore as ms
from mindspore import ops,nn
import sys

class clientAVG(Client):
    def __init__(self,args,id,train_sample,test_sample,**kwargs):
        super().__init__(args,id,train_sample,test_sample,**kwargs)
    
    def train(self):
        print(self.id,end=' ')
        if self.id==9:
            print('')
        sys.stdout.flush() 
        trainloader_all = self.load_train_data_minibatch(iterations=1)
        

        self.model.set_train()

        max_local_epochs = self.local_epochs

        train_losses = 0

        for step in range(max_local_epochs):
            for trainloader in trainloader_all:
                for x,y in trainloader:
                    loss = self.train_step(x,y)
                    train_losses += loss.asnumpy() * y.shape[0]


