import copy
import math
import os

import ujson
import h5py

from flcore.clients.clientalidpfl import clientALIDPFL
from flcore.optimizer.utils.RDP.get_max_steps import get_max_steps
from flcore.servers.serverbase import Server

from utils.data_utils import read_server_testset
from flcore.optimizer.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis
import time

import mindspore
from mindspore import nn,ops
import mindspore as ms

def compute_new_tau(mu, C, Gamma, sigma, d, hat_B, Rs, Rc, tau_star):

    T = min(Rs * tau_star, Rc)
    print(f"mu={mu}, C={C}, Gamma={Gamma}, sigma={sigma}, d={d}, "
          f"hat_B={hat_B}, Rs={Rs}, Rc={Rc}, T={T}, tau_star={tau_star}")
    dp_noise_bound = (sigma ** 2 * C ** 2 * d) / (hat_B ** 2)

    molecule = (4 / (mu ** 2)) + 3 * (C ** 2) + 2 * Gamma * T * mu + dp_noise_bound

    denominator = (2 + 2 / T) * (C ** 2 + dp_noise_bound)
    ret = math.sqrt(1 + molecule / (denominator + 1e-6))
    print(f"分子 = {molecule}, 分母 = {denominator}, 原始tau = {ret}")
    ret = int(ret + 0.5)  
    ret = max(1, ret)
    ret = min(ret, 100) 
    return ret

def sub_model(model_1, model_2):
    ret_model = copy.deepcopy(model_1)
    for param_tensor in model_1.parameters_dict():
        sub_param = model_1.parameters_dict()[param_tensor] - model_2.parameters_dict()[param_tensor]
        ret_model.parameters_dict()[param_tensor].set_data(sub_param.copy())
    return ret_model

def compute_l2_norm_of_model(model):
    l2_norm = 0
    for params_tensor in model.parameters_dict():
        l2_norm += ops.norm(model.parameters_dict()[params_tensor].value())**2
    return l2_norm ** .5

def compute_mu(grad_l2_lists, model_l2_lists, weights):
    mu = 0.0
    for grad_l2, model_l2, w in zip(grad_l2_lists, model_l2_lists, weights):
        mu += w * grad_l2 / (model_l2 + 1e-6)
    return mu



class ALIDPFL(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientALIDPFL)
        self.rs_server_acc = []  
        self.rs_server_loss = []  
        self.loss = nn.CrossEntropyLoss()  
        self.batch_sample_ratio = args.batch_sample_ratio
        self.dp_sigma = args.dp_sigma  
        self.dp_norm = args.dp_norm  
        self.need_adaptive_tau = args.need_adaptive_tau
        self.tau_star = args.local_iterations  
        self.rs_tau_list = [self.tau_star]  
        self.dp_epsilon = args.dp_epsilon

        self.global_rounds = args.global_rounds
        self.local_iterations = args.local_iterations

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.hat_B = int(self.batch_sample_ratio * min([client.train_samples for client in self.clients]))
        self.dimension_of_model = sum(ops.numel(param) for name,param in self.global_model.parameters_dict().items())

        delta = 10 ** (-5)
        orders = (list(range(2, 64)) + [128, 256, 512])  

        self.Rc = get_max_steps(self.dp_epsilon, delta, self.batch_sample_ratio, self.dp_sigma, orders)

        print(f"===================== Rc={self.Rc} =====================")

        if self.need_adaptive_tau and self.global_rounds >= self.Rc:  
            self.need_adaptive_tau = False
            self.local_iterations = 1
            self.tau_star = 1
            for client in self.clients:
                client.need_adaptive_tau = False
                self.local_iterations = 1
                self.tau_star = 1

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "./results/"

        current_path = os.path.abspath(__file__)  
        parent_directory = os.path.dirname(current_path)  
        parent_directory = os.path.dirname(parent_directory)  
        parent_directory = os.path.dirname(parent_directory) 
        root_directory = os.path.dirname(parent_directory)  
        config_json_path = root_directory + "\\" + self.dataset + "\\config.json"

        if not os.path.exists(result_path):
            os.makedirs(result_path)


        if len(self.rs_test_acc):
            
            algo = algo + "_" + self.goal+"_Rs_"+str(self.global_rounds)   
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            extra_msg = f"dataset = {self.dataset}, learning_rate = {self.learning_rate},\n" \
                        f"rounds = {self.global_rounds}, batch_sample_ratio = {self.batch_sample_ratio},\n" \
                        f"num_clients = {self.num_clients}, algorithm = {self.algorithm} \n" \
                        f"have_PD = {self.args.privacy}, dp_sigma = {self.args.dp_sigma}\n" \
                        f"epsilon = {self.dp_epsilon}, dp_norm = {self.dp_norm}\n" \
                        f"Rs = {self.global_rounds}, Rc = {self.Rc}" \
                        f"need_adaptive_tau = {self.need_adaptive_tau}"
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
                hf.create_dataset('rs_tau_list', data=self.rs_tau_list[:-1])  
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_server_acc', data=self.rs_server_acc)
                hf.create_dataset('rs_server_loss', data=self.rs_server_loss)
                hf.create_dataset('extra_msg', data=extra_msg, dtype=h5py.string_dtype(encoding='utf-8'))    

    def evaluate_server(self,q=0.2,test_batch_size=64):
        test_loader_full = read_server_testset(self.dataset, q=q, batch_size=test_batch_size)
        self.global_model.set_train(False)

        test_acc = 0
        test_num = 0

        for x, y in test_loader_full:
            output = self.global_model(x)
            label = y.astype(ms.int32)
            loss = self.loss(output,label)

            test_acc += (output.argmax(1) == y).asnumpy().sum()
            test_num += y.shape[0]

        accuracy = test_acc / test_num
        self.rs_server_acc.append(accuracy)
        
        self.rs_server_loss.append(loss.item())
        print("Accuracy at server: {:.4f}".format(accuracy))
        print("Loss at server: {:.4f}".format(loss.item()))

    def send_models(self):  # sever->client
        assert (len(self.clients) > 0)

        for client in self.clients:
             client.set_tau(self.tau_star)  

             client.set_parameters(self.global_model)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        if self.need_adaptive_tau:
            self.global_model.set_train(False)
            for client in self.clients:
                client.model.set_train(False)
        
        self.global_model = copy.deepcopy(self.uploaded_models[0])

        for param_tensor in self.global_model.parameters_dict():
            zeor_tensor = ops.zeros_like(self.global_model.parameters_dict()[param_tensor],dtype = ms.float32 )
            self.global_model.parameters_dict()[param_tensor].set_data(zeor_tensor)

        self.add_parameters(self.uploaded_weights,self.uploaded_models)

        if self.need_adaptive_tau:
            model_diff_l2_list = []
            grad_diff_l2_list = []
            loss_diff_list = []
            for client in self.clients:
                model_diff = sub_model(client.model, self.global_model)
                model_diff_l2_list.append(compute_l2_norm_of_model(model_diff))
                grad_diff = sub_model(client.grad_client_model, client.grad_global_model)
                grad_diff_l2_list.append(compute_l2_norm_of_model(grad_diff))
                loss_diff_list.append(client.loss_client_model - client.loss_global_model)
            
            self.mu_strong_convex = compute_mu(grad_diff_l2_list, model_diff_l2_list, self.uploaded_weights)
            self.Gamma = abs(sum([w * loss_diff for w, loss_diff in zip(self.uploaded_weights, loss_diff_list)]))

            self.tau_star = compute_new_tau(
                mu=self.mu_strong_convex,
                C=self.dp_norm,
                Gamma=self.Gamma,
                sigma=self.dp_sigma,
                d=self.dimension_of_model,
                hat_B=self.hat_B,
                Rs=self.global_rounds,
                Rc=self.Rc,
                tau_star=self.tau_star
            )
        
        if self.need_adaptive_tau:
            self.rs_tau_list.append(self.tau_star)
        else:
            self.rs_tau_list.append(self.local_iterations)
    
    def train(self):
        for i in range(0, self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()  
            
            if i % self.eval_gap == 0:  
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model by personalized")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
            
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)  
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if i % self.eval_gap == 0:  
                print("\nEvaluate global model by global")
                self.evaluate_server(q=0.2, test_batch_size=64)
            
            print("adap_tau_list: ", self.rs_tau_list[:-1])
            if sum(self.rs_tau_list) > self.Rc:
                print(f"Rc={self.Rc} is running out")
                break

        if sum(self.rs_tau_list) <= self.Rc:
            print(f"Rs={self.global_rounds} is running out")
        
        print("Best local_avg_accuracy={:.4f}, Last local_avg_accuracy={:.4f}".format(
            max(self.rs_test_acc), self.rs_test_acc[-1]))
        print("Best server_accuracy={:.4f}, Last server_accuracy={:.4f}".format(
            max(self.rs_server_acc), self.rs_server_acc[-1]))
        print("Last server_loss={:.4f}".format(self.rs_server_loss[-1]))
        print("Average time cost per round={:.4f}".format(sum(self.Budget[1:]) / len(self.Budget[1:])))

        self.save_global_model()
        self.save_results()

        
        
       
