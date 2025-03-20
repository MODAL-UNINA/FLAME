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


# Load data
def load_data(zero_day, client_id, batch_size):
    data_path = f"../mydata/CICDDoS2019/zero_day_{zero_day}/client_{client_id}/"
    train_data = torch.load(os.path.join(data_path, "train_data.pt"))
    test_data = torch.load(os.path.join(data_path, "test_data.pt"))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    num_examples = {"trainset" : len(train_data), "testset" : len(test_data)}
    
    return train_loader, test_loader, num_examples


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
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    power_draw = result.stdout.decode('utf-8').strip().split('\n')[device]
    print(f"Consumo energetico: {power_draw} W")
    return power_draw

def plot_loss_and_accuracy_history(loss_history, accuracy_history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Loss
    ax[0].plot(range(1, len(loss_history) + 1), loss_history, color='blue')
    ax[0].set_xlabel("Round")
    ax[0].set_ylabel("Loss")
    ax[0].set_title(f"Training Loss for Client {client_id}")

    # Accuracy
    ax[1].plot(range(1, len(accuracy_history) + 1), accuracy_history, color='green')
    ax[1].set_xlabel("Round")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title(f"Test Accuracy for Client {client_id}")

    plt.tight_layout()
    plt.show()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model: Net, trainloader: torch.utils.data.DataLoader, 
                 testloader: torch.utils.data.DataLoader, num_examples: Dict) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        
        self.best_jsd_value = float('inf')

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

        # Prevent division by zero.
        global_sum = np.sum(np.abs(global_flat))
        local_sum = np.sum(np.abs(local_flat))

        if global_sum == 0 or local_sum == 0:
            print("Warning: One of the arrays sums to zero, returning inf for JSD.")
            return float('inf')

        # The constant epsilon is added to prevent zero values.
        epsilon = 1e-10
        global_prob = np.abs(global_flat) + epsilon
        local_prob = np.abs(local_flat) + epsilon

        # Normalized to a probability distribution.
        global_prob /= np.sum(global_prob)
        local_prob /= np.sum(local_prob)

        jsd_value = jensenshannon(global_prob, local_prob)

        return np.mean(jsd_value) if jsd_value.size > 0 else 0.0


    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        global_parameters = self.get_parameters(config={})

        # Initialize
        best_weights = global_parameters

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

        # Flatten
        local_parameters_flat = [param.flatten() for param in local_parameters]
        best_weights_flat = [param.flatten() for param in best_weights]

        packed_parameters = local_parameters_flat + best_weights_flat

        return (
            packed_parameters, 
            self.num_examples["trainset"], 
            {
            "current_jsd_value": current_jsd_value,
            },
        )


    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, accuracy, f1 = test(self.model, self.testloader, device=DEVICE)

        test_loss_history.append(float(loss))
        test_accuracy_history.append(float(accuracy))
        test_f1_history.append(float(f1))

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


# Defining the profiler
num_clients = 12
time_list = []
train_loss_history = []
test_loss_history = []
test_accuracy_history = []
test_f1_history = []
EPOCHS = 5
batch_size = 32
device = 0
server_IP = '127.0.0.1:8080'
output_file = 'output_flame'
zero_day = 'NTP'
client_id = 0     # Open a new interactive window and change the id to run the client, i.e. if there are 12 clients, change the id from 0 to 11.

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

t1 =  time.perf_counter()

model = Net()
model.to(DEVICE)

print("Load data")
trainloader, testloader, num_examples = load_data(zero_day, client_id, batch_size)

# Start client
client = FlowerClient(model, trainloader, testloader, num_examples).to_client()
fl.client.start_client(server_address=server_IP, client=client)
plot_loss_and_accuracy_history(train_loss_history, test_accuracy_history)


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
    print(f"PARAMS: {params}") # k=migliaia
    print(f"flops: {flops}") # M=milioni
    print(f"MACS: {macs}") # MMACs = milioni di moltiplicazioni-accumuli.
    average_time = np.mean(time_list[1:])
    print(f'Tempi per client {client_id}: ', average_time)


# convert to float
flops = float(flops[:-2])
macs = float(macs[:-5])
params = float(params[:-1])
t2 =  time.perf_counter()
execution_time = t2 - t1
print("time: ", execution_time)
rss, vms, data, shared, text, lib, dirty, uss, pss, swap, gpu_memory = memory_usage(process, device=CVD)
power_draw = get_gpu_power_consume(device=CVD)
power_draw = float(power_draw)

os.makedirs(f'../results/CICDDoS2019/{output_file}/Results_{zero_day}', exist_ok=True)
with open(f'../results/CICDDoS2019/{output_file}/Results_{zero_day}/Client{client_id}_performance.txt', 'w') as f:
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

# save loss and accuracy
D = {}
D['train_loss_history'] = train_loss_history
D['test_loss_history'] = test_loss_history
D['test_accuracy_history'] = test_accuracy_history
D['test_f1_history'] = test_f1_history

with open(f'../results/CICDDoS2019/{output_file}/Results_{zero_day}/Client{client_id}_performance.pkl', 'wb') as f:
    pickle.dump(D, f)
