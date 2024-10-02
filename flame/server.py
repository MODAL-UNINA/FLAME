#%%
import numpy as np
import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from CustomStrategy import Dis

import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#%%
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

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
  
#%%
# Start Server
num_clients = 12
n_round = 100

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
# weights
weights_dict = model.state_dict()
# convert to list of numpy arrays
weights = [val.numpy() for _, val in weights_dict.items()]
# serialize arrays to Parameters
parameters = fl.common.ndarrays_to_parameters(weights)

# Start Flower server
history =  fl.server.start_server(
  server_address = "0.0.0.0:8080",
  config = fl.server.ServerConfig(num_rounds=n_round),
  strategy = Dis(
                    fraction_fit=1.,  # Sample 10% of available clients for the next round
                    min_fit_clients=num_clients,  # Minimum number of clients to be sampled for the next round. Minimum number of clients used during training.
                    # settare min_fit_clients <= min available clients
                    min_available_clients=num_clients,  # Minimum number of clients that need to be connected to the server before a training round can start. Minimum number of total clients in the system
                    initial_parameters=parameters,
                    evaluate_metrics_aggregation_fn=weighted_average))

# %%
# plot
def plot_loss_accuracy_server(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    round, measures = zip(*history.losses_distributed)
    plt.plot(round, measures, label="loss distributed")
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    round, measures = zip(*history.metrics_distributed['accuracy'])
    plt.plot(round, measures, label="accuracy distributed", color='orange')
    plt.xlabel("round")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


plot_loss_accuracy_server(history)

#%%
