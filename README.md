# ALI-DPFL


---

**paper arxiv：** https://arxiv.org/abs/2308.10457

**paper title：**ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations

**Abstract：**Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks. We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication round are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DP-SGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of $\textbf{D}$ifferentially $\textbf{P}$rivate $\textbf{F}$ederated $\textbf{L}$earning with$ \textbf{A}$daptive $\textbf{L}$ocal $\textbf{I}$terations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and CIFAR10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. 

## Preparation

### Download Dependencies

```
pip install -r requirements.txt
```

### Generate Datasets

code for MNIST, Fashion-mnist and Cifar10 is already, please run `ALI_DPFL_mindspore/dataset/generate_XX.py` to generate datasets online.

IID or Non-IID can be adjust by 

```python
niid = True/False
balance = False/True
partition = 'dir'/None
```

set parameter `need_server_testset = True` to support test global model in server.

## Get Started

run `ALI-DPFL\system\main_alidpfl.py` to get start.
