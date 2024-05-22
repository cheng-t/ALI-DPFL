from flcore.optimizer.dp_optimizer import DPSGD
from flcore.clients.clientbase import Client
import mindspore as ms
from mindspore import nn,ops
import numpy as np
import sys

class clientDPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.dp_norm = args.dp_norm
        self.batch_sample_ratio = args.batch_sample_ratio
        self.auto_s = args.auto_s
        self.local_iterations = args.local_iterations

        if self.privacy:
            self.optimizer = DPSGD(
                l2_norm_clip=self.dp_norm,  
                noise_multiplier=self.dp_sigma,
                minibatch_size=self.batch_size,  # batch_size
                microbatch_size=1,  
                params=self.model.trainable_params(),
                learning_rate=self.learning_rate,
            )
        
    def train(self):

        print(self.id,end=' ')
        if self.id==9:
            print('')
        sys.stdout.flush()
        minibatch_size = int(self.train_samples * self.batch_sample_ratio)
        trainloader_all = self.load_train_data_minibatch(minibatch_size=minibatch_size,
                                                     iterations=self.local_iterations)
        self.model.set_train()


        max_local_epochs = self.local_epochs  
        for step in range(max_local_epochs):  

            for trainloader in trainloader_all:
                for x,y in trainloader:
                    gradients_list = []
                    losses_list = []
                    for x_single,y_single in zip(x,y):
                        x_single = ops.expand_dims(x_single,0)
                        y_single = ops.expand_dims(y_single,0)

                        loss,grad = self.train_step_dpsgd(x_single,y_single)
                        gradients_list.append(grad)
                        losses_list.append(loss)
                    self.optimizer(gradients_list)
                    self.loss_batch_avg = sum(losses_list) / len(losses_list)  
                   
                

