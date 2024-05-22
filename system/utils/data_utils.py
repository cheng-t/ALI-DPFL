import random

import numpy as np
import os

from mindspore.dataset import GeneratorDataset

import mindspore as ms

def read_data(dataset, idx, is_train=True):
    if is_train:
        current_directory = os.path.dirname(__file__)       

        train_data_dir = os.path.join(current_directory,'..','..', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'  
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()  

        return train_data

    else:
        current_directory = os.getcwd()
        test_data_dir = os.path.join(current_directory, dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = ms.Tensor(train_data['x']).type(ms.float32)
        y_train = ms.Tensor(train_data['y']).type(ms.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]  
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = ms.Tensor(test_data['x']).type(ms.float32)
        y_test = ms.Tensor(test_data['y']).type(ms.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = ms.Tensor(X_train).type(ms.int64)
        X_train_lens = ms.Tensor(X_train_lens).type(ms.int64)
        y_train = ms.Tensor(train_data['y']).type(ms.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = ms.Tensor(X_test).type(ms.int64)
        X_test_lens = ms.Tensor(X_test_lens).type(ms.int64)
        y_test = ms.Tensor(test_data['y']).type(ms.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = ms.Tensor(train_data['x']).type(ms.int64)
        y_train = ms.Tensor(train_data['y']).type(ms.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = ms.Tensor(test_data['x']).type(ms.int64)
        y_test = ms.Tensor(test_data['y']).type(ms.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_server_testset(dataset, q=0.2, batch_size=64):

    if dataset not in ["mnist", "Cifar10", "fmnist"]:
        print("read_server_testset还没处理非mnist, Cifar10外的数据集，请检查")
        print("此函数位于data_utils.py,函数名为read_server_testset()")
        print("即将强制终止程序,祝您debug好运...")
        exit(0)
    current_directory = os.getcwd()
    test_data_dir = os.path.join(current_directory, dataset, 'test/')

    test_file = test_data_dir + "server_testset" + '.npz'
    with open(test_file, 'rb') as f:
        test_data = np.load(f, allow_pickle=True)['data'].tolist()
    X_test = ms.Tensor(test_data['x']).type(ms.float32)
    y_test = ms.Tensor(test_data['y']).type(ms.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]

    num_elements = int(len(test_data) * q)

    test_data = random.sample(test_data, num_elements)
    dataloader = GeneratorDataset(test_data,column_names=['x','y'],shuffle=True)
    dataloader = dataloader.batch(batch_size,drop_remainder=True)
    return dataloader
   


