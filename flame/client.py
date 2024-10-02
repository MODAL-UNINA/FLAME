#%%
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import psutil
import os
import time
import subprocess
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import GPUtil
import argparse
from flops_profiler.profiler import get_model_profile
import matplotlib.pyplot as plt
import pickle
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#%%
class Net(nn.Module):
  def __init__(self) -> None:
    super(Net, self).__init__()
    self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
    self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
    self.pool = nn.MaxPool1d(kernel_size=2)
    self.fc1 = nn.Linear(384 * 1, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 1)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 384 * 1)
    x = F.sigmoid(self.fc1(x))
    return x

def train(    
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    client_id: int,
    ):
  """Train the model on the training set."""
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)

  epoch_loss = []
  for epoch in range(epochs):
    batch_loss = []

    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        log_probs = net(inputs)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
    epoch_loss.append(sum(batch_loss)/len(batch_loss))

  running_loss = sum(epoch_loss)/len(epoch_loss)
  print("[%d] epoch loss: %.3f" % (epoch + 1, running_loss))

  return running_loss


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,):
  """Validate the model on the test set."""
  #criterion = torch.nn.CrossEntropyLoss()
  correct, test_loss = 0, 0.0
  all_pred = []
  all_target = []

  with torch.no_grad():
      for i, (inputs, labels) in enumerate(testloader):
          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = net(inputs)
          test_loss += F.binary_cross_entropy(outputs, labels, reduction='sum').item()
          predicted = (outputs > 0.5).float()
          correct += predicted.eq(labels.data.view_as(predicted)).long().cpu().sum()
          all_pred.append(predicted)
          all_target.append(labels)

      test_loss /= len(testloader.dataset)
      accuracy = 100.00 * correct / len(testloader.dataset)

      all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
      all_target = torch.cat(all_target, dim=0).cpu().numpy()
      
      f1 = f1_score(all_target.flatten(), all_pred.flatten())*100
      
  print(f"Test Accuracy: {accuracy}%")
  print(f"Test F1: {f1}%")
  return test_loss, accuracy, f1


# 加载本地文件夹中的数据
def load_data(zero_day, client_id, batch_size):
    data_path = f"/home/modal-workbench/Projects/Pian/mydata/CICDDoS2019/zero_day_{zero_day}/client_{client_id}/"
    train_data = torch.load(os.path.join(data_path, "train_data.pt"))
    test_data = torch.load(os.path.join(data_path, "test_data.pt"))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    num_examples = {"trainset" : len(train_data), "testset" : len(test_data)}
    
    return train_loader, test_loader, num_examples

#TO DO: class_weights implement

def memory_usage(process=None, device=0):
    if process is None:
        process = psutil.Process(os.getpid())
    print('Process ID:', process.pid)
    memory_info = process.memory_full_info()
    memory_usage_bytes = memory_info.rss 
    vmemory_usage_bytes = memory_info.vms
    processed_data = memory_info.data
    shared_memory = memory_info.shared
    text_memory = memory_info.text
    lib_memory = memory_info.lib
    dirty_memory = memory_info.dirty
    uss = memory_info.uss
    pss = memory_info.pss
    swap = memory_info.swap

    gpus = GPUtil.getGPUs()
    gpus = gpus[device]
    gpus_memory = gpus.memoryUsed
    print(f"GPU memory used: {gpus_memory} MB")

    return memory_usage_bytes / (1024 ** 2), vmemory_usage_bytes / (1024 ** 2), processed_data / (1024 ** 2), shared_memory / (1024 ** 2), text_memory / (1024 ** 2), lib_memory / (1024 ** 2), dirty_memory / (1024 ** 2), uss / (1024 ** 2), pss / (1024 ** 2), swap / (1024 ** 2), gpus_memory
    # Converti in MB

def get_gpu_power_consume(device=0):
    # Esegui il comando nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)

    # Estrai l'output
    power_draw = result.stdout.decode('utf-8').strip().split('\n')[device]
    print(f"Consumo energetico: {power_draw} W")
    return power_draw

def plot_loss_and_accuracy_history(loss_history, accuracy_history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制 Loss 子图
    ax[0].plot(range(1, len(loss_history) + 1), loss_history, color='blue')
    ax[0].set_xlabel("Round")
    ax[0].set_ylabel("Loss")
    ax[0].set_title(f"Training Loss for Client {client_id}")

    # 绘制 Accuracy 子图
    ax[1].plot(range(1, len(accuracy_history) + 1), accuracy_history, color='green')
    ax[1].set_xlabel("Round")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title(f"Test Accuracy for Client {client_id}")

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()

def plot_loss_and_f1_history(loss_history, accuracy_history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制 Loss 子图
    ax[0].plot(range(1, len(loss_history) + 1), loss_history, color='blue')
    ax[0].set_xlabel("Round")
    ax[0].set_ylabel("Loss")
    ax[0].set_title(f"Test Loss for Client {client_id}")

    # 绘制 Accuracy 子图
    ax[1].plot(range(1, len(accuracy_history) + 1), accuracy_history, color='green')
    ax[1].set_xlabel("Round")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title(f"Test F1 for Client {client_id}")

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()

#%%
# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model: Net, trainloader: torch.utils.data.DataLoader, 
                 testloader: torch.utils.data.DataLoader, num_examples: Dict) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        
        self.best_jsd_value = float('inf')  # 初始化

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def compute_jsd(self, global_parameters: List[np.ndarray], local_parameters: List[np.ndarray]) -> float:
        global_param = global_parameters[-2]
        local_param = local_parameters[-2]

        global_flat = global_param.flatten()
        local_flat = local_param.flatten()

        # 防止空数组或全零情况导致除零错误
        global_sum = np.sum(np.abs(global_flat))
        local_sum = np.sum(np.abs(local_flat))

        if global_sum == 0 or local_sum == 0:
            print("Warning: One of the arrays sums to zero, returning inf for JSD.")
            return float('inf')

        # 取绝对值并加上小常数 epsilon 防止零值
        epsilon = 1e-10
        global_prob = np.abs(global_flat) + epsilon
        local_prob = np.abs(local_flat) + epsilon

        # 归一化为概率分布
        global_prob /= np.sum(global_prob)
        local_prob /= np.sum(local_prob)

        # # 检查概率分布
        # print("Global probabilities:", global_prob)
        # print("Local probabilities:", local_prob)

        # 计算 JSD
        jsd_value = jensenshannon(global_prob, local_prob)

        # # 检查 JSD 值
        # print("JSD values:", jsd_value)

        return np.mean(jsd_value) if jsd_value.size > 0 else 0.0

    def compute_cosine_similarity(self, global_parameters: List[np.ndarray], local_parameters: List[np.ndarray]) -> float:
        global_param = global_parameters[-2]
        local_param = local_parameters[-2]

        global_flat = global_param.flatten()
        local_flat = local_param.flatten()

        # 防止空数组或全零情况导致除零错误
        if np.linalg.norm(global_flat) == 0 or np.linalg.norm(local_flat) == 0:
            print("Warning: One of the arrays is all zeros, returning NaN for cosine similarity.")
            return float('nan')

        # 计算余弦相似度
        cosine_sim = 1 - distance.cosine(global_flat, local_flat)

        return cosine_sim

    def compute_chebyshev_distance(self, global_parameters: List[np.ndarray], local_parameters: List[np.ndarray]) -> float:
        global_param = global_parameters[-2]
        local_param = local_parameters[-2]

        global_flat = global_param.flatten()
        local_flat = local_param.flatten()

        # 检查空数组或全零情况
        if global_flat.size == 0 or local_flat.size == 0:
            print("Warning: One of the arrays is empty, returning NaN for Chebyshev distance.")
            return float('nan')

        # 计算切比雪夫距离
        chebyshev_dist = distance.chebyshev(global_flat, local_flat)

        return chebyshev_dist

    def compute_euclidean_distance(self, global_parameters: List[np.ndarray], local_parameters: List[np.ndarray]) -> float:
        global_param = global_parameters[-2]
        local_param = local_parameters[-2]

        global_flat = global_param.flatten()
        local_flat = local_param.flatten()

        # 检查空数组或全零情况
        if global_flat.size == 0 or local_flat.size == 0:
            print("Warning: One of the arrays is empty, returning NaN for Euclidean distance.")
            return float('nan')

        # 计算欧氏距离
        euclidean_dist = distance.euclidean(global_flat, local_flat)

        return euclidean_dist

    def compute_minkowski_distance(self, global_parameters: List[np.ndarray], local_parameters: List[np.ndarray], p: float) -> float:
        global_param = global_parameters[-2]
        local_param = local_parameters[-2]

        global_flat = global_param.flatten()
        local_flat = local_param.flatten()

        # 检查空数组或全零情况
        if global_flat.size == 0 or local_flat.size == 0:
            print("Warning: One of the arrays is empty, returning NaN for Minkowski distance.")
            return float('nan')

        if global_flat.size != local_flat.size:
            print("Warning: Arrays are of different lengths, returning NaN for Minkowski distance.")
            return float('nan')

        # 计算闵可夫斯基距离，支持自定义p值
        minkowski_dist = distance.minkowski(global_flat, local_flat, p)

        return minkowski_dist


    def compute_mahalanobis_distance(self, global_parameters: List[np.ndarray], local_parameters: List[np.ndarray]) -> float:
        global_param = global_parameters[-2]
        local_param = local_parameters[-2]

        global_flat = global_param.flatten()
        local_flat = local_param.flatten()

        # 检查空数组或长度不同
        if global_flat.size == 0 or local_flat.size == 0:
            print("Warning: One of the arrays is empty, returning NaN for Mahalanobis distance.")
            return float('nan')

        if global_flat.size != local_flat.size:
            print("Warning: Arrays are of different lengths, returning NaN for Mahalanobis distance.")
            return float('nan')

        # 计算协方差矩阵
        data = np.vstack([global_flat, local_flat])
        cov_matrix = np.cov(data.T)  # 协方差矩阵
        
        # 使用伪逆来代替逆矩阵，防止协方差矩阵为奇异矩阵
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        # 计算马氏距离
        mahalanobis_dist = distance.mahalanobis(global_flat, local_flat, inv_cov_matrix)

        return mahalanobis_dist


    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        global_parameters = self.get_parameters(config={})

        # 初始化或更新最佳权重
        best_weights = global_parameters  # 首次将最佳权重设置为当前全局模型的权重

        print("Training local model")
        t0 = time.perf_counter()
        running_loss = train(self.model, self.trainloader, epochs=EPOCHS, device=DEVICE, client_id=client_id)
        
        train_loss_history.append(float(running_loss))
        t1 = time.perf_counter()
        time_list.append(t1-t0)
        print("Training done")

        local_parameters = self.get_parameters(config={})

        current_jsd_value = self.compute_jsd(global_parameters, local_parameters)
        print(f"current_jsd_value is: {current_jsd_value}") 

        if current_jsd_value < self.best_jsd_value:
            self.best_jsd_value = current_jsd_value
            best_weights = local_parameters


        # 展平参数
        local_parameters_flat = [param.flatten() for param in local_parameters]
        best_weights_flat = [param.flatten() for param in best_weights]

        # 将展平后的参数打包成列表
        packed_parameters = local_parameters_flat + best_weights_flat  # 将两个列表合并

        return (
            packed_parameters, 
            self.num_examples["trainset"], 
            {
            "current_jsd_value": current_jsd_value,  # 示例值
            },
        )


    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, accuracy, f1 = test(self.model, self.testloader, device=DEVICE)

        test_loss_history.append(float(loss))
        test_accuracy_history.append(float(accuracy))
        test_f1_history.append(float(f1))

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


#%%
# Defining the profiler
num_clients = 12
time_list = []
train_loss_history = []
test_loss_history = []
test_accuracy_history = []
test_f1_history = []
EPOCHS = 1
batch_size = 32
device = 0
server_IP = '127.0.0.1:8080'
output_file = 'output_f_m_0.1_epoch_1'
# 'MSSQL', 'UDPLag', 'NetBIOS', 'NTP', 'Syn',
# 'DNS', 'LDAP', 'Portmap', 'TFTP', 'SSDP', 'SNMP', 'UDP', 'WebDDoS'
zero_day = 'NTP'
client_id = 11

_parser = argparse.ArgumentParser(prog="client", description="Run the client.",)
_parser.add_argument('--client_id', type=int, default=client_id)
_parser.add_argument('--device', type=int, default=device)
_parser.add_argument('--server_IP', type=str, default=server_IP)


args = _parser.parse_known_args()[0]
client_id = args.client_id
server_IP = args.server_IP
CVD = args.device
DEVICE: str = torch.device(f"cuda:{CVD}" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# show this process
pid_client = os.getpid()
process = psutil.Process(os.getpid())
print(process.pid, pid_client)
#%%
t1 =  time.perf_counter()

model = Net()
model.to(DEVICE)

print("Load data")
trainloader, testloader, num_examples = load_data(zero_day, client_id, batch_size)

# Start client
client = FlowerClient(model, trainloader, testloader, num_examples).to_client()
fl.client.start_client(server_address=server_IP, client=client)

plot_loss_and_accuracy_history(train_loss_history, test_accuracy_history)

plot_loss_and_f1_history(test_loss_history, test_f1_history)

#%%
with torch.cuda.device(CVD):
    flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(batch_size, 28, 1), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None, # the list of modules to ignore in the profiling
                                    func_name='forward') # the function name to profile, "forward" by default, for huggingface generative models, `generate` is used
    print(f'DEVICE: {DEVICE}')
    print(f"PARAMS: {params}") #k=migliaia
    print(f"flops: {flops}") #M=milioni
    print(f"MACS: {macs}") #MMACs = milioni di moltiplicazioni-accumuli.
    average_time = np.mean(time_list[1:])
    print(f'Tempi per client {client_id}: ', average_time)

    '''
    All MACs are FLOPs, but not all FLOPs are MACs. 
    FLOPs include all floating point operations, 
    while MACs focus specifically on multiplication and accumulation. 
    In many deep learning applications, multiplication and accumulation are the main operations, 
    and MACs are often a better representation of the computational complexity of the model than FLOPs.
    '''

# %%
# convert to float
flops = float(flops[:-2])
macs = float(macs[:-5])
params = float(params[:-1])
t2 =  time.perf_counter()
execution_time = t2 - t1
print("time: ", execution_time)
# time.sleep(10)
# print(f'Memory Usage ov {client_id}:', memory_usage(process, device=CVD))
rss, vms, data, shared, text, lib, dirty, uss, pss, swap, gpu_memory = memory_usage(process, device=CVD)
power_draw = get_gpu_power_consume(device=CVD)
power_draw = float(power_draw)

os.makedirs(f'{output_file}/Results_{zero_day}', exist_ok=True)
with open(f'{output_file}/Results_{zero_day}/Client{client_id}_performance.txt', 'w') as f:
    f.write(f'Client: {client_id}\n')
    f.write(f'PID: {pid_client}\n')
    f.write(f'GPU: {CVD}\n')
    f.write(f'params: {params} k\n')
    f.write(f'flops: {flops} M\n')
    f.write(f'MACS: {macs} MMACs\n')
    f.write(f'Execution time mean: {average_time} seconds\n')
    f.write(f'Memory Usage RSS: {rss} MB\n') # RAM resident set size
    f.write(f'Memory Usage VMS: {vms} MB\n') 
    f.write(f'Memory Usage DATA: {data} MB\n')
    f.write(f'Memory Usage SHARED: {shared} MB\n')
    f.write(f'Memory Usage TEXT: {text} MB\n') 
    f.write(f'Memory Usage LIB: {lib} MB\n') 
    f.write(f'Memory Usage DIRTY: {dirty} MB\n') 
    f.write(f'Memory Usage USS: {uss} MB\n') 
    f.write(f'Memory Usage PSS: {pss} MB\n') 
    f.write(f'Memory Usage SWAP: {swap} MB\n') 
    f.write(f'GPU Memory Usage: {gpu_memory} MB\n') # totale se ci sono client sulla stessa macchina
    f.write(f'GPU Power Consumption: {power_draw} W\n')
    f.write('\n')
    f.write(f'Number of examples: {num_examples}\n')
    f.write(f'THROUGHPUT: {flops/execution_time} FLOPS\n')
    f.write(f'ENERGY EFFICIENCY: {flops/execution_time/power_draw} FLOPS/W\n')
#%%
#%%

# save loss and accuracy
D = {}
D['train_loss_history'] = train_loss_history
D['test_loss_history'] = test_loss_history
D['test_accuracy_history'] = test_accuracy_history
D['test_f1_history'] = test_f1_history

with open(f'{output_file}/Results_{zero_day}/Client{client_id}_performance.pkl', 'wb') as f:
    pickle.dump(D, f)
