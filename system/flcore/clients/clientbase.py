import mindspore as ms
import copy
import sys

import numpy as np
from utils.data_utils import read_client_data
from mindspore import nn,ops
from mindspore.dataset import GeneratorDataset
import mindspore.dataset as ds



class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.minibatch_size = minibatch_size    
        self.iterations = iterations            

    def generator(self):

        for i in range(self.iterations):                                
            indices = np.where(np.random.rand(len(self.dataset)) < (self.minibatch_size / len(self.dataset)))[0]

            if indices.size > 0:
                yield indices

    def get_dataset(self):
        indices_all = self.generator()
        re_dataloader = []
        for i in range(self.iterations):
            indices = next(indices_all)
            pseudo_batch_size = indices.size
            
            dataloader = GeneratorDataset(self.dataset, sampler=indices , column_names=['x','y'])
            dataloader = dataloader.batch(pseudo_batch_size)
            re_dataloader.append(dataloader)

        return re_dataloader


class Client():
    """
    Base class for clients in federated learning.
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset  # str
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs       

        self.mini_batch_size = 0 

        # check BatchNorm
        self.has_BatchNorm = False  
        for cell_name, cell in self.model.cells_and_names():  

            if isinstance(cell, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = nn.SGD(self.model.trainable_params(), learning_rate=self.learning_rate)

    def __lt__(self,other):
        if self.id<other.id:
            return True
        else:
            return False

    def load_train_data(self,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        dataloader = GeneratorDataset(train_data,shuffle=True,column_names=["x",'y'])
        dataloader = dataloader.batch(batch_size,drop_remainder=True)
        return dataloader

    def load_train_data_minibatch(self, minibatch_size=None, iterations=None):
        if minibatch_size is None:
            minibatch_size = self.batch_size
        if iterations is None:
            iterations = 1
        train_data = read_client_data(self.dataset, self.id, is_train=True)   
        sampler = IIDBatchSampler(train_data,minibatch_size,iterations)

        return sampler.get_dataset() 

    
    def load_test_data(self,batch_size=None):
        if batch_size==None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset,self.id,is_train=False)
        dataloader = GeneratorDataset(test_data,shuffle=True,column_names=['x','y'])
        dataloader = dataloader.batch(batch_size,drop_remainder=False)
        return dataloader
    
    def forward_fn(self,data,label):
        logits = self.model(data)
        label = label.astype(ms.int32)
        loss = self.loss_fn(logits,label)
        return loss, logits
    
    def train_step(self,data,label):
        grad_fn = ms.value_and_grad(self.forward_fn,None,self.optimizer.parameters,has_aux=True)
        (loss,_),grads = grad_fn(data,label)
        self.optimizer(grads)
        return loss
    
    def train_step_dpsgd(self,data,label):
        grad_fn = ms.value_and_grad(self.forward_fn,None,self.optimizer.parameters,has_aux=True)
        (loss,_),grads = grad_fn(data,label)
        return loss,grads

    def train_metrics(self):
        trainloader = self.load_train_data()
    

        train_num = 0
        losses = 0
        self.model.set_train()
        
        for x,y in trainloader:
            loss = self.train_step(x,y)
            losses += loss.asnumpy() *y.shape[0]
            train_num += y.shape[0]

        return losses,train_num
    
    def test_metric(self):
        testloader = self.load_test_data()
        self.model.set_train(False)

        test_acc = 0
        test_num = 0
        test_loss = 0
        for x,y in testloader:
            pred = self.model(x)
            label = y.astype(ms.int32)
            loss = self.loss_fn(pred,label)
            test_loss += loss.asnumpy()

            test_num += y.shape[0]
            test_acc += (pred.argmax(1) == y).asnumpy().sum()
        
        test_acc /= test_num
        auc = 0

        return test_acc,test_num,auc,test_loss
    
    def set_parameters(self,model):

        for param_tensor in model.parameters_dict():

            param_tensor_copy = model.parameters_dict()[param_tensor].copy()

            self.model.parameters_dict()[param_tensor].set_data(param_tensor_copy)

