import os
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    return x

def binarize_y(y, labels_list):
    '''
    Objective:
        Binarize the target variable
    Inputs:
        y: target variable
        label: label to binarize
    Output:
        binarized target variable
    '''
    y_binary = y.copy()
    # y_binary[y_binary != label] = 'Attack'
    y_binary = np.where(y_binary.isin(labels_list), 1, 0)
    return y_binary



def create_zero_day(df, label):
    '''
    Objective:
        Create a zero day attack
    Inputs:
        df: dataframe
        label: label to create the zero day attack
        seed: random seed
    Output:
        dataframe with the zero day attack
    '''
    zero_day = df[df['Label'] == label]
    df = df[df['Label'] != label]
    return df, zero_day


def zero_day_simulation(dataframe, label):

    df, zero_day = create_zero_day(dataframe, label)
    all_attack_labels = dataframe['Label'].unique().tolist()
    all_attack_labels.remove(label)
    all_attack_labels.remove('Benign')
    all_attack_labels = sorted(all_attack_labels, key=lambda x: x == 'WebDDoS')

    assert len(all_attack_labels) == 12

    return df, zero_day, all_attack_labels


# load the data
path = '../FLAME/data_raw/'
dspaths = []
for dirname, _, filenames in os.walk(path + 'DATASET'):
    for filename in filenames:
        if filename.endswith('.pkl'):
            pds = os.path.join(dirname, filename)
            dspaths.append(pds)
            print(pds)

dspaths = sorted(dspaths)

individual_dfs = [pd.read_pickle(dsp) for dsp in tqdm(dspaths)]
df = pd.concat(individual_dfs, axis=0, ignore_index=True)

# Preprocessing
df.isna().sum().sum()
df['Flow Packets/s'] = df['Flow Packets/s'].fillna(df['Flow Packets/s'].mean())

# remove the 'DrDoS_' from the labels
df['Label'] = df['Label'].apply(lambda x: x.replace('DrDoS_', '') if 'DrDoS_' in x else x)
# merge the labels of the 'UDPLag' and 'UDP-lag' attacks
df['Label'] = df['Label'].apply(lambda x: 'UDPLag' if x == 'UDP-lag' else x)

best_features = ['Down/Up Ratio', 'URG Flag Count', 'Bwd Packet Length Min', 'CWE Flag Count',
                    'Avg Fwd Segment Size', 'Fwd Packet Length Mean', 'Min Packet Length',
                    'Fwd Packet Length Min', 'Packet Length Mean', 'Bwd Packet Length Mean']

df_best = df[best_features + ['Label']]

# remove collinear features
df_best_80 = remove_collinear_features(df_best.drop(columns=['Label']), 0.8)

df_best_80['Label'] = df_best['Label']
df_best_80.drop(columns=['Label']).corr()


df_sampled_80 = df_best_80.copy()
# check duplicates
df_sampled_80.duplicated().sum()
# remove duplicates
df_sampled_80 = df_sampled_80.drop_duplicates()


# plot the distribution of the features for each attack type
# 'MSSQL', 'UDPLag', 'NetBIOS', 'NTP', 'Syn', 'DNS', 'LDAP', 'Portmap', 'TFTP', 'SSDP', 'SNMP', 'UDP', 'WebDDoS'
data_begin = df_sampled_80[df_sampled_80['Label'] == 'Benign']
data_mssql = df_sampled_80[df_sampled_80['Label'] == 'MSSQL']
data_udplag = df_sampled_80[df_sampled_80['Label'] == 'UDPLag']
data_netbios = df_sampled_80[df_sampled_80['Label'] == 'NetBIOS']
data_ntp = df_sampled_80[df_sampled_80['Label'] == 'NTP']
data_syn = df_sampled_80[df_sampled_80['Label'] == 'Syn']
data_dns = df_sampled_80[df_sampled_80['Label'] == 'DNS']
data_ldap = df_sampled_80[df_sampled_80['Label'] == 'LDAP']
data_portmap = df_sampled_80[df_sampled_80['Label'] == 'Portmap']
data_tftp = df_sampled_80[df_sampled_80['Label'] == 'TFTP']
data_ssdp = df_sampled_80[df_sampled_80['Label'] == 'SSDP']
data_snmp = df_sampled_80[df_sampled_80['Label'] == 'SNMP']
data_udp = df_sampled_80[df_sampled_80['Label'] == 'UDP']
data_webddos = df_sampled_80[df_sampled_80['Label'] == 'WebDDoS']


fig, ax = plt.subplots(3, 2, figsize=(15, 6))
features = ['Down/Up Ratio', 'URG Flag Count', 'Bwd Packet Length Min', 'CWE Flag Count',
            'Avg Fwd Segment Size','Bwd Packet Length Mean']

for i in range(6):
    ax[i//2, i%2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.4f}'.format(x)))
    sns.kdeplot(data_mssql[features[i]], ax=ax[i//2, i%2], label='MSSQL')
    sns.kdeplot(data_udplag[features[i]], ax=ax[i//2, i%2], label='UDPLag')
    sns.kdeplot(data_netbios[features[i]], ax=ax[i//2, i%2], label='NetBIOS')
    sns.kdeplot(data_ntp[features[i]], ax=ax[i//2, i%2], label='NTP')
    sns.kdeplot(data_syn[features[i]], ax=ax[i//2, i%2], label='Syn')
    sns.kdeplot(data_dns[features[i]], ax=ax[i//2, i%2], label='DNS')
    sns.kdeplot(data_ldap[features[i]], ax=ax[i//2, i%2], label='LDAP')
    sns.kdeplot(data_portmap[features[i]], ax=ax[i//2, i%2], label='Portmap')
    sns.kdeplot(data_tftp[features[i]], ax=ax[i//2, i%2], label='TFTP')
    sns.kdeplot(data_ssdp[features[i]], ax=ax[i//2, i%2], label='SSDP')
    sns.kdeplot(data_snmp[features[i]], ax=ax[i//2, i%2], label='SNMP')
    sns.kdeplot(data_udp[features[i]], ax=ax[i//2, i%2], label='UDP')
    sns.kdeplot(data_webddos[features[i]], ax=ax[i//2, i%2], label='WebDDoS')
plt.legend(loc='center left', bbox_to_anchor=(1, 1.0))
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()


# create the clients data
client_train = dict()
client_val = dict()
client_test = dict()

# define the label to be the zero day attack
which_label = 'WebDDoS'
how_many = len(df_sampled_80[df_sampled_80['Label'] == which_label])

df_sampled, zero_day, all_attack_labels = zero_day_simulation(df_sampled_80, which_label)

# random sampling 100 benign samples as the test set
benign_test = df_sampled[df_sampled['Label'] == 'Benign'].sample(100, replace=False, random_state=42)
df_sampled = df_sampled.drop(benign_test.index)

# random sampling 100 attack samples as the test set
if how_many > 100:
    attack_test = zero_day.sample(100, replace=False, random_state=42)
else:
    attack_test = zero_day

testset = pd.concat([benign_test, attack_test], axis=0)
testset = testset.reset_index(drop=True)
testset['Label'] = binarize_y(testset['Label'], [which_label])


# creat train and test set for each client
i = -1
for label in tqdm(all_attack_labels, desc='Creating clients'):
    
    i += 1

    if i != len(all_attack_labels):

        malicious_sampled_train = df_sampled[df_sampled['Label'] == label]
        df_sampled = df_sampled.drop(malicious_sampled_train.index)

        benign_sampled = df_sampled[df_sampled['Label'] == 'Benign'].sample(1027, replace=False, random_state=42)    # len(df_sampled['Label'] == 'Benign') / clients_num = 1027
        df_sampled = df_sampled.drop(benign_sampled.index)

        client_train[i] = pd.concat([malicious_sampled_train, benign_sampled], axis=0)
        client_train[i] = client_train[i].reset_index(drop=True)

        # set the label to binary
        client_train[i]['Label'] = binarize_y(client_train[i]['Label'], [label])

        # for test set
        client_test[i] = testset.copy()

        # scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(client_train[i].drop(columns=['Label']).values)
        X_test = scaler.transform(client_test[i].drop(columns=['Label']).values)

        # create the tensor dataset
        client_train[i] = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(2), torch.tensor(client_train[i]['Label'].values, dtype=torch.float32).unsqueeze(1))
        client_test[i] = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(2), torch.tensor(client_test[i]['Label'].values, dtype=torch.float32).unsqueeze(1))

        print('Finished label {}'.format(label))

    else:
        break


# save the client_train data
with open('../FLAME/data/client_trains_{}.pkl'.format(which_label), 'wb') as f:
    pickle.dump(client_train, f)

# save the client_test data
with open('../FLAME/data/client_test_{}.pkl'.format(which_label), 'wb') as f:
    pickle.dump(client_test, f)