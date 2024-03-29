import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import copy
import pickle
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import psutil


# local update 
class LocalUpdate(object):
    def __init__(self, dataset_train, class_weights):
        self.loss_func = nn.BCELoss(weight=torch.tensor(class_weights).to(device))
        self.ldr_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
        epoch_loss = []
        for iter in range(1):
            batch_loss = []

            for batch_idx, (inputs, labels) in enumerate(self.ldr_train):
                inputs = inputs.to(device)
                labels = labels.to(device)

                net.zero_grad()
                log_probs = net(inputs)
                loss = self.loss_func(log_probs, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss)/len(epoch_loss)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 1)
        x = F.sigmoid(self.fc1(x))
        return x


def fedavg(w, weight_scalling_factor_list):
    w_global = copy.deepcopy(w[0])
    for i in w_global.keys():
        w_global[i] = w_global[i] * weight_scalling_factor_list[0]

    for k in w_global.keys():
        for i in range(1, len(w)):
            w_global[k] += w[i][k] * weight_scalling_factor_list[i]
    return w_global


def test(net, dataset_test):
    net.eval()
    # testing
    correct = 0

    all_pred = []
    all_target = []

    ldr_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(ldr_test):
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            y_pred = (output > 0.5).float()
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            all_pred.append(y_pred)
            all_target.append(target)

        accuracy = 100.00 * correct / len(ldr_test.dataset)

        # f1_score, precision, recall
        all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
        all_target = torch.cat(all_target, dim=0).cpu().numpy()
        
        f1 = f1_score(all_target.flatten(), all_pred.flatten())*100
        precision = precision_score(all_target.flatten(), all_pred.flatten())*100
        recall = recall_score(all_target.flatten(), all_pred.flatten())*100

    return accuracy, f1, precision, recall


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# zero day attack
zero_day = 'Syn'

# load the client data
with open('../FLAME/data/client_trains_{}.pkl'.format(zero_day), 'rb') as f:
    client_train = pickle.load(f)

# # load the test data
with open('../FLAME/data/client_test_{}.pkl'.format(zero_day), 'rb') as f:
    client_test = pickle.load(f)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net_glob = CNN().to(device)
net_glob.train()

# training
loss_train = []
total_test_accuracy = [] 
total_test_f1 = []
total_test_precision = []
total_test_recall = []

net_best = None
best_acc = None

for iter in range(100):    
    loss_locals = []
    w_locals = []

    accuracy_list_test = []
    f1_list_test = []
    precision_list_test = []
    recall_list_test = []

    # calculate the weighting factor of clients
    clients_trn_data = {i: client_train[i] for i in range(len(client_train))}
    data_list = [len(clients_trn_data[i]) for i in range(len(clients_trn_data))]
    weight_scalling_factor_list = [i/sum(data_list) for i in data_list]

    # for each client, train the model
    for idx in range(len(client_train)): 

        dataset_train = client_train[idx]
        dataset_test = client_test[idx]

        # calculate the class distribution, and assign weights to the classes
        num_attack = torch.sum(dataset_train[:,-1][1].type(torch.LongTensor)).item()
        num_benign = len(dataset_train[:,-1][1]) - num_attack
        class_weights = [num_benign/num_attack]

        # test global model before training
        net_glob.eval()
        acc, test_loss, f1, precision, recall = test(net_glob, dataset_test)

        # collect test results for each client
        accuracy_list_test.append(acc)
        f1_list_test.append(f1)
        precision_list_test.append(precision)
        recall_list_test.append(recall)

        # local update
        local = LocalUpdate(dataset_train, class_weights=class_weights)
        w, loss = local.train(net=copy.deepcopy(net_glob).to(device))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))

    # save the best model
    if best_acc == None:
        best_acc = sum(accuracy_list_test) / len(accuracy_list_test)
        net_best = copy.deepcopy(net_glob)
    else:
        if (sum(accuracy_list_test) / len(accuracy_list_test)) > best_acc:
            best_acc = sum(accuracy_list_test) / len(accuracy_list_test)
            net_best = copy.deepcopy(net_glob)  

    # update global weights
    w_glob = fedavg(w_locals, weight_scalling_factor_list)
    net_glob.load_state_dict(w_glob)

    # collect test results for each round
    total_test_accuracy.append(accuracy_list_test)
    total_test_f1.append(f1_list_test)
    total_test_precision.append(precision_list_test)
    total_test_recall.append(recall_list_test)

    # print train loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average Train Loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)     # collect train loss

    # print test results
    print(f' Average Test Accuracy: {sum(accuracy_list_test) / len(accuracy_list_test):.2f}%',
            f' Average F1: {sum(f1_list_test) / len(f1_list_test):.2f}%',
            f' Average Precision: {sum(precision_list_test) / len(precision_list_test):.2f}%',
            f' Average Recall: {sum(recall_list_test) / len(recall_list_test):.2f}%')

# memory usage
memory_used = psutil.Process().memory_info().rss
memory_cost = memory_used / 1024 / 1024
print(f'Memory used: {memory_cost:.2f} MB')

# calculate the best model's test results
model = net_best
model.eval()

total_cm = []
acc_list= []
f1_list = []
precision_list = []
recall_list = []

# test the model on each client data
for k in range(len(client_train)):
    testset = client_test[k]
    all_pred = []
    all_target = []
    correct = 0
    ldr_test = DataLoader(testset, batch_size=32, shuffle=False)
    for idx, (data, target) in enumerate(ldr_test):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        y_pred = (output > 0.5).float()
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        all_pred.append(y_pred)
        all_target.append(target)

    acc = 100.00 * correct / len(ldr_test.dataset)

    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
    all_target = torch.cat(all_target, dim=0).cpu().numpy()

    f1 = f1_score(all_target, all_pred)*100
    precision = precision_score(all_target, all_pred)*100
    recall = recall_score(all_target, all_pred)*100

    acc_list.append(acc)
    f1_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)

    cm = confusion_matrix(all_target, all_pred)
    total_cm.append(cm)

print(f'Average accuracy: {np.mean(acc_list):.2f}%, f1: {np.mean(f1_list):.2f}%, precision: {np.mean(precision_list):.2f}%, recall: {np.mean(recall_list):.2f}%')

# plot the confusion matrix
fig, ax = plt.subplots(2, 6, figsize=(16, 5))
for i in range(2):
    for j in range(6):
        sns.heatmap(total_cm[i*6+j], annot=True, fmt="d", cmap='Blues', ax=ax[i][j])
        # let fontsize of the numbers in the heatmap be bigger
        for t in ax[i][j].texts:
            t.set_text(t.get_text())
            t.set_fontsize(20)
        ax[i][j].set_title(f'Client {i*6+j}', fontsize=15)
        ax[i][j].set_xlabel('Predicted', fontsize=15)
        ax[i][j].set_ylabel('True', fontsize=15)
# set spacing between subplots
plt.tight_layout(pad=2)
# set overall title
plt.suptitle('Confusion Matrix - {}'.format(zero_day), fontsize=20, y=1.05)
plt.show()
