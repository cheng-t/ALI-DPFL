import numpy as np
import mindspore as ms
import numpy as np
from utils.data_utils import read_client_data
import copy
from random import sample
import os

class Server():
    def __init__(self,args):
        self.args = args
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)  
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio  # Ratio of clients per round,args.join_ratio = 1.0 default
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm  
        self.save_folder_name = args.save_folder_name
        self.goal = args.goal
        self.eval_gap = args.eval_gap
        self.clients = []
        self.selected_clients = []



        self.rs_test_acc = []  
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        

    def set_clients(self,clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,i,len(train_data),len(test_data))
            self.clients.append(client)
       
    def select_clients(self):

        selected_clients = sample(self.clients,self.num_join_clients)

        self.current_num_join_clients = self.num_join_clients

        return sorted(selected_clients)


    
    # sever->client
    def send_models(self):
        
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)


    def aggregate_parameters(self):

        assert(len(self.selected_clients)>0)

        for param_tensor in self.global_model.parameters_dict():

            param_tensor_avg = sum(c.model.parameters_dict()[param_tensor] for c in self.selected_clients)/len(self.selected_clients)

            param_tensor_avg_copy = param_tensor_avg.copy()

            self.global_model.parameters_dict()[param_tensor].set_data(param_tensor_avg_copy)

    def evaluate(self,acc=None,loss=None):

        acc_sum = 0
        test_loss = 0
        for client in self.clients:
            test_acc,test_num,auc,test_loss = client.test_metric()
            acc_sum += test_acc
            test_loss += test_loss
        
        if acc == None:
            self.rs_test_acc.append(acc_sum/self.num_clients)
        else:
            acc.append(acc_sum/self.num_clients)

        if loss == None:
            self.rs_train_loss.append(test_loss/self.num_clients)
        else:
            loss.append(test_loss/self.num_clients)
        
        return acc_sum/self.num_clients,test_loss

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):  
            self.uploaded_weights[i] = w / tot_samples
        
    def add_parameters(self, weight, client_model):
        for param_tensor in self.global_model.parameters_dict():
            param_tensor_sum = sum(model.parameters_dict()[param_tensor]*w for model,w in zip(client_model,weight))
            self.global_model.parameters_dict()[param_tensor].set_data(param_tensor_sum.copy())

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".ckpt")
        ms.save_checkpoint(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))

        param_dict = ms.load_checkpoint(model_path)
        param_not_load, _ = ms.load_param_into_net(self.global_model, param_dict)
        if param_not_load == []:
            print("Load successful")