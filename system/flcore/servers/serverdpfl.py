import numpy as np
from flcore.clients.clientdpfl import clientDPFL
from flcore.servers.serverbase import Server

from mindspore import nn,ops
import sys
import mindspore as ms
import h5py
import os
import ujson

from utils.data_utils import read_server_testset
from flcore.optimizer.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis

class DPFL(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientDPFL)
        self.rs_server_acc = []  
        self.rs_server_loss = []  
        self.loss_fn = nn.CrossEntropyLoss()  
        self.batch_sample_ratio = args.batch_sample_ratio
        self.dp_sigma = args.dp_sigma  

        self.global_rounds = args.global_rounds
        self.local_iterations = args.local_iterations

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"

        current_path = os.path.abspath(__file__)  
        parent_directory = os.path.dirname(current_path)  
        parent_directory = os.path.dirname(parent_directory) 
        parent_directory = os.path.dirname(parent_directory) 
        root_directory = os.path.dirname(parent_directory)  
        config_json_path = root_directory + "\\dataset\\" + self.dataset + "\\config.json"

        if not os.path.exists(result_path):
            os.makedirs(result_path)


        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
        eps, opt_order = apply_dp_sgd_analysis(q=self.batch_sample_ratio,
                                               sigma=self.dp_sigma,
                                              
                                               steps=self.global_rounds * self.local_iterations,
                                               orders=orders,
                                               delta=10e-5)
        print("eps:", format(eps) + "| order:", format(opt_order))

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)  
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            extra_msg = f"dataset = {self.dataset}, learning_rate = {self.learning_rate},\n" \
                        f"rounds = {self.global_rounds}, batch_sample_ratio = {self.batch_sample_ratio},\n" \
                        f"num_clinets = {self.num_clients}, algorithm = {self.algorithm} \n" \
                        f"have_PD = {self.args.privacy}, dp_sigma = {self.args.dp_sigma}\n" \
                        f"epsilon = {eps}\n"
            with open(config_json_path) as f:
                data = ujson.load(f)

            extra_msg = extra_msg + "--------------------config.json------------------------\n" \
                                    "num_clients={}, num_classes={}\n" \
                                    "non_iid={}, balance={},\n" \
                                    "partition={}, alpha={}\n".format(
                data["num_clients"], data["num_classes"], data["non_iid"],
                data["balance"], data["partition"], data["alpha"])

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_server_acc', data=self.rs_server_acc)
                hf.create_dataset('rs_server_loss', data=self.rs_server_loss)
                hf.create_dataset('extra_msg', data=extra_msg, dtype=h5py.string_dtype(encoding='utf-8'))


    def evaluate_server(self, q=0.2, test_batch_size=64):        

        test_loader_full = read_server_testset(self.dataset, q=q, batch_size=test_batch_size)
        self.global_model.set_train(False)

        test_acc = 0
        test_num = 0
        test_loss = 0
        i=0 

        for x,y in test_loader_full:
            pred = self.global_model(x)
            label = y.astype(ms.int32)     

            loss = self.loss_fn(pred,label)
            test_loss += loss.asnumpy()   
            i+=1

            test_num += y.shape[0]
            test_acc += (pred.argmax(1) == y).asnumpy().sum()
        
        accuracy = test_acc / test_num
        loss = test_loss/i

        print("Accuracy at server: {:.4f}".format(accuracy))
        print("Loss at server: {:.4f}".format(loss))


        
    
    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()

            print(i,':',end = '') 
            sys.stdout.flush()
            for client in self.selected_clients:
                client.train()    

            self.aggregate_parameters()

            self.send_models()
                       
            if i % self.eval_gap == 0:  
                print('')
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model by personalized")
                acc,loss = self.evaluate()
                print(f'Global model round {i} accuracy: {acc}')
                print(f'loss {loss}')

                print("\nEvaluate global model by global")
                self.evaluate_server(q=0.2, test_batch_size=64)

    def test(self):
        acc,loss = self.evaluate()
        print('Training Finished')
        print(f'Personalized Final Accuracy: {acc}')
        print(f'Personalized Final Loss: {loss}')
        
        print('Global')
        self.evaluate_server(q=0.2, test_batch_size=64)

