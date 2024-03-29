import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
import copy
import pickle
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import psutil

class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = list(range(len(self.dataset[0])))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[0][self.idxs[item]], self.dataset[1][self.idxs[item]]
        return image, label


# local update for fedbr
class LocalUpdate_FedBR(object):
    def __init__(self, dataset_train, class_weights):
        self.loss_func = nn.BCELoss(weight=torch.tensor(class_weights).to(device))
        self.ldr_train = DataLoader(dataset_train, batch_size=8, shuffle=True, drop_last=True)

        self.discriminator = Discriminator().to(device)

        self.if_updated = True
        self.all_global_unlabeled_z = None


    def sim(self, x1, x2):
        return torch.cosine_similarity(x1, x2, dim=1)


    def train(self, net, unlabeled=None):
        net.train()

        disc_opt = torch.optim.SGD(self.discriminator.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
        gen_opt = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)

        mu = 0.5
        lam = 1.0
        gamma = 1.0
        tau1 = 2.0
        tau2 = 2.0

        epoch_loss = []
        for iter in range(1):
            batch_loss = []
            for batch_idx, (input_data, labels) in enumerate(self.ldr_train):
                input_data, labels = input_data.to(device), labels.to(device)

                all_x = input_data
                all_y = labels

                all_unlabeled = torch.cat([x for x in unlabeled])

                q = torch.ones((len(all_y), 1)) / 1   
                q = q.to(device)
                
                if self.if_updated:
                    self.original_feature = net(all_x)[1].clone().detach()
                    self.original_classifier = net(all_x)[0].clone().detach()
                    self.all_global_unlabeled_z = net(all_unlabeled)[1].clone().detach()
                    self.if_updated = False

                all_unlabeled_z = net(all_unlabeled)[1]
                all_self_z = net(all_x)[1]

                embedding1 = self.discriminator(all_unlabeled_z.clone().detach())
                embedding2 = self.discriminator(self.all_global_unlabeled_z)
                embedding3 = self.discriminator(all_self_z.clone().detach())

                disc_loss = torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))
                disc_loss = gamma * torch.sum(disc_loss) / len(embedding1)

                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step()

                embedding1 = self.discriminator(all_unlabeled_z)
                embedding2 = self.discriminator(self.all_global_unlabeled_z)
                embedding3 = self.discriminator(all_self_z)

                disc_loss = - torch.log(torch.exp(self.sim(embedding1, embedding2) * tau1) / (torch.exp(self.sim(embedding1, embedding2) * tau1) + torch.exp(self.sim(embedding1, embedding3) * tau2)))
                disc_loss = torch.sum(disc_loss) / len(embedding1)
                
                all_preds = net.classifier(all_self_z)
                classifier_loss =torch.mean(torch.sum(F.binary_cross_entropy_with_logits(all_preds, all_y, reduction='none'), 1))
                aug_penalty =torch.mean(torch.sum(torch.mul(F.binary_cross_entropy_with_logits(all_preds, all_y, reduction='none'), q), 1))

                gen_loss =  classifier_loss + (mu * disc_loss) + lam * aug_penalty

                disc_opt.zero_grad()
                gen_opt.zero_grad()
                gen_loss.backward()
                gen_opt.step()

                batch_loss.append(classifier_loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input = nn.Linear(32, 16)
        self.hiddens = nn.ModuleList([
            nn.Linear(16, 16)
            for _ in range(1)])
        self.output = nn.Linear(16, 8)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.featurizer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.featurizer(x)
        x0 = x.view(len(x), -1)
        x = x.view(-1, 32 * 1)
        x = self.classifier(x)

        return x, x0


# model aggregation
def fedbr(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# mean for pseudo data
def get_augmentation_mean_data(clients, weights):

    augmentation_data = []
    
    for i in range(8):
        chosen_client = torch.randint(0, len(clients), (1,))
        client_data, client_weights = clients[chosen_client]
        indexs = torch.randint(0, len(client_data[0]), (10,))
    
        current_aug_data = torch.zeros_like(client_data[0][0])
        current_aug_data = current_aug_data.unsqueeze(0)
        for index in indexs:
            current_aug_data += client_data[0][index]/len(indexs)
        augmentation_data.append(current_aug_data.to(device))

    return augmentation_data

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
            output, _ = net(data)
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
    
    # RSM for pseudo data
    weights = [1.0 / 12] * 12 
    clients = list(zip(clients_trn_data.values(), weights))
    uda_device = get_augmentation_mean_data(clients, weights)

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
        local = LocalUpdate_FedBR(dataset_train, class_weights=class_weights)
        w, loss = local.train(net=copy.deepcopy(net_glob).to(device), unlabeled=uda_device)
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
    w_glob = fedbr(w_locals)
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
        output, _ = model(data)
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
