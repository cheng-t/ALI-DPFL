import mindspore as ms
from mindspore import dataset as ds
import mindspore.dataset.transforms as transforms
from mindspore.dataset import GeneratorDataset
from download import download
import mindspore.dataset.vision as vision
import numpy as np
import os

from utils.dataset_utils import check, separate_data, split_data, save_file
from generate_server_testset import generate_server_testset

import random

dir_path = "Cifar10/"


def download_cifar10(path):
    cifar10_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
    download(cifar10_url, path, kind="tar.gz", replace=False)

def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, need_server_testset=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
    
    download_cifar10(dir_path+"rawdata")

    transform = transforms.Compose([vision.ToTensor(), vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),is_hwc=False)])

    trainset = ds.Cifar10Dataset(dir_path+'rawdata/cifar-10-batches-bin',usage = 'train');
    trainset.map(operations=transform)
    testset = ds.Cifar10Dataset(dir_path+'rawdata/cifar-10-batches-bin',usage = 'test');
    testset.map(operations=transform)

    trainloader = GeneratorDataset(trainset,column_names=['x','y'],shuffle=False);
    trainloader.batch(len(trainset))
    testloader = GeneratorDataset(testset,column_names=['x','y'],shuffle=False);
    testloader.batch(len(testset))

    i = 0
    x_all = []
    y_all = []
    for x,y in trainloader:

        x_all.append(np.transpose(x,(2,0,1)).asnumpy())
        y_all.append(y.asnumpy())

    for x,y in testloader:

        x_all.append(np.transpose(x,(2,0,1)).asnumpy())
        y_all.append(y.asnumpy())

    dataset_image = np.array(x_all)
    dataset_label = np.array(y_all)



    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                        niid, balance, partition)
    train_data, test_data = split_data(X, y)

    
    if need_server_testset:
        generate_server_testset(test_data, test_path)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)




random.seed(1)
np.random.seed(1)
num_clients = 10
num_classes = 10

if __name__ == "__main__":
    niid = True
    balance = False
    partition = 'dir'
    need_server_testset = True

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, need_server_testset)