import numpy as np
import time
import copy

from flcore.clients.clientbase import Client
from flcore.optimizer.dp_optimizer import DPSGD
from mindspore import ops
import mindspore as ms
import sys

class clientALIDPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.dp_norm = args.dp_norm
        self.batch_sample_ratio = args.batch_sample_ratio
        self.auto_s = args.auto_s
        self.local_iterations = args.local_iterations
        self.need_adaptive_tau = args.need_adaptive_tau
        self.tau_star = args.local_iterations

        self.loss_global_model = 1.0
        self.grad_global_model = 1.0
        self.loss_client_model = 1.0
        self.grad_client_model = 1.0

        if self.privacy:
            self.optimizer = DPSGD(
                l2_norm_clip=self.dp_norm,  
                noise_multiplier=self.dp_sigma,
                minibatch_size=self.batch_size,  
                microbatch_size=1,  

                params=self.model.trainable_params(),
                learning_rate=self.learning_rate,
            )

    def set_tau(self, tau_star):
        self.tau_star = tau_star

    def train(self):
        print("---------------------------------------")
        print(f"Client {self.id} is training, privacy={self.privacy}, AUTO-S={self.auto_s}")
        minibatch_size = int(self.train_samples * self.batch_sample_ratio)
        trainloader_all = self.load_train_data_minibatch(minibatch_size=minibatch_size,
                                                     iterations=self.tau_star)

        self.model.set_train()
        
        max_local_epochs = self.local_epochs  
        for step in range(max_local_epochs):  

            i=0
            for trainloader in trainloader_all:
                len_y = 0
                for x,y in trainloader:
                    len_y = len(y)
                print(f"Clint {self.id} 的第 {i + 1} 次 iteration, 本次采样个数len(y): {len_y}")
                i+=1
                sys.stdout.flush()

                for x,y in trainloader:

                    if self.need_adaptive_tau:
                        global_model = copy.deepcopy(self.model)
                        loss,grad = self.train_step_dpsgd(x,y)
                        self.loss_global_model = loss.item()
            
                        for param_name,param_grad in zip(global_model.parameters_dict(),grad):
                           
                            global_model.parameters_dict()[param_name].set_data(copy.deepcopy(param_grad))

                        self.grad_global_model = copy.deepcopy(global_model)

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

                    if self.need_adaptive_tau:
                        
                        client_model = copy.deepcopy(self.model)
                        loss,grad = self.train_step_dpsgd(x,y)
                        self.loss_client_model = loss.item()
                        for param_name,param_grad in zip(client_model.parameters_dict(),grad):
                            client_model.parameters_dict()[param_name].set_data(copy.deepcopy(param_grad))
 
                        self.grad_client_model = copy.deepcopy(client_model)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  