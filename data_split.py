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

seed = 0
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
# remove the 'DrDoS_' from the labels
df['Label'] = df['Label'].apply(lambda x: x.replace('DrDoS_', '') if 'DrDoS_' in x else x)
# merge the labels of the 'UDPLag' and 'UDP-lag' attacks
df['Label'] = df['Label'].apply(lambda x: 'UDPLag' if x == 'UDP-lag' else x)

# check duplicates
df.duplicated().sum()

# remove duplicates
df = df.drop_duplicates()

# remove NAN
df.isna().sum().sum()
df = df.dropna()

# resort
df = df.reset_index(drop=True) 

# drop useless features
df = df.drop(columns=['Flow ID', 'Protocol', 'Timestamp', 'Bwd PSH Flags', 
                      'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 
                      'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 
                      'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 
                      'Bwd Avg Bulk Rate', 'FIN Flag Count', 'PSH Flag Count', 'ECE Flag Count'])

# keep useful features
total_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
       'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
       'Fwd Packet Length Max', 'Fwd Packet Length Min',
       'Fwd Packet Length Mean', 'Fwd Packet Length Std',
       'Bwd Packet Length Max', 'Bwd Packet Length Min',
       'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
       'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
       'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
       'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
       'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
       'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
       'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
       'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
       'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'URG Flag Count',
       'CWE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
       'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
       'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
       'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
       'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
       'Idle Std', 'Idle Max', 'Idle Min']

df[total_features] = df[total_features].astype('float32') 

df_best = df[total_features + ['Label']]

# remove collinear features
df_best_80 = remove_collinear_features(df_best.drop(columns=['Label']), 0.8)

df_best_80['Label'] = df_best['Label']
df_best_80.drop(columns=['Label']).corr()


df_best_80.isna().sum().sum()
df_best_80 = df_best_80.dropna()  

# resort
df_best_80 = df_best_80.reset_index(drop=True)

df_sampled_80 = df_best_80.copy()


# # plot the distribution of the features for each attack type
# # 'MSSQL', 'UDPLag', 'NetBIOS', 'NTP', 'Syn', 'DNS', 'LDAP', 'Portmap', 'TFTP', 'SSDP', 'SNMP', 'UDP', 'WebDDoS'
# data_begin = df_sampled_80[df_sampled_80['Label'] == 'Benign']
# data_mssql = df_sampled_80[df_sampled_80['Label'] == 'MSSQL']
# data_udplag = df_sampled_80[df_sampled_80['Label'] == 'UDPLag']
# data_netbios = df_sampled_80[df_sampled_80['Label'] == 'NetBIOS']
# data_ntp = df_sampled_80[df_sampled_80['Label'] == 'NTP']
# data_syn = df_sampled_80[df_sampled_80['Label'] == 'Syn']
# data_dns = df_sampled_80[df_sampled_80['Label'] == 'DNS']
# data_ldap = df_sampled_80[df_sampled_80['Label'] == 'LDAP']
# data_portmap = df_sampled_80[df_sampled_80['Label'] == 'Portmap']
# data_tftp = df_sampled_80[df_sampled_80['Label'] == 'TFTP']
# data_ssdp = df_sampled_80[df_sampled_80['Label'] == 'SSDP']
# data_snmp = df_sampled_80[df_sampled_80['Label'] == 'SNMP']
# data_udp = df_sampled_80[df_sampled_80['Label'] == 'UDP']
# data_webddos = df_sampled_80[df_sampled_80['Label'] == 'WebDDoS']


# fig, ax = plt.subplots(3, 2, figsize=(15, 6))
# features = ['Down/Up Ratio', 'URG Flag Count', 'Bwd Packet Length Min', 'CWE Flag Count',
#             'Avg Fwd Segment Size','Bwd Packet Length Mean']

# for i in range(6):
#     ax[i//2, i%2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.4f}'.format(x)))
#     sns.kdeplot(data_mssql[features[i]], ax=ax[i//2, i%2], label='MSSQL')
#     sns.kdeplot(data_udplag[features[i]], ax=ax[i//2, i%2], label='UDPLag')
#     sns.kdeplot(data_netbios[features[i]], ax=ax[i//2, i%2], label='NetBIOS')
#     sns.kdeplot(data_ntp[features[i]], ax=ax[i//2, i%2], label='NTP')
#     sns.kdeplot(data_syn[features[i]], ax=ax[i//2, i%2], label='Syn')
#     sns.kdeplot(data_dns[features[i]], ax=ax[i//2, i%2], label='DNS')
#     sns.kdeplot(data_ldap[features[i]], ax=ax[i//2, i%2], label='LDAP')
#     sns.kdeplot(data_portmap[features[i]], ax=ax[i//2, i%2], label='Portmap')
#     sns.kdeplot(data_tftp[features[i]], ax=ax[i//2, i%2], label='TFTP')
#     sns.kdeplot(data_ssdp[features[i]], ax=ax[i//2, i%2], label='SSDP')
#     sns.kdeplot(data_snmp[features[i]], ax=ax[i//2, i%2], label='SNMP')
#     sns.kdeplot(data_udp[features[i]], ax=ax[i//2, i%2], label='UDP')
#     sns.kdeplot(data_webddos[features[i]], ax=ax[i//2, i%2], label='WebDDoS')
# plt.legend(loc='center left', bbox_to_anchor=(1, 1.0))
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.5)
# plt.show()


# create the clients data
client_train = dict()
client_val = dict()
client_test = dict()

# define the label to be the zero day attack
which_label = 'WebDDoS'
how_many = 1350

df_sampled, zero_day, all_attack_labels = zero_day_simulation(df_sampled_80, which_label)


i = -1
for label in tqdm(all_attack_labels, desc='Creating clients'):
    i += 1
    if i != len(all_attack_labels):
        if i == 11:
            malicious_sampled_train = df_sampled[df_sampled['Label'] == label]
        else:
            malicious_sampled_train = df_sampled[df_sampled['Label'] == label].sample(7650, replace=False, random_state=seed)

        df_sampled = df_sampled.drop(malicious_sampled_train.index)

        benign_sampled = df_sampled[df_sampled['Label'] == 'Benign'].sample(9000, replace=False, random_state=seed)
        df_sampled = df_sampled.drop(benign_sampled.index)

        # random split benign data
        benign_train, benign_test = train_test_split(benign_sampled, test_size=0.15, random_state=seed)

        client_train[i] = pd.concat([malicious_sampled_train, benign_train], axis=0)
        client_train[i] = client_train[i].reset_index(drop=True)

        # set the label to binary
        client_train[i]['Label'] = binarize_y(client_train[i]['Label'], [label])

        # for test set
        zero_day_sampled = zero_day.sample(how_many, random_state=seed, replace=False)
        zero_day = zero_day.drop(zero_day_sampled.index)

        X_test = zero_day_sampled.drop(columns=['Label']).values
        y_test = zero_day_sampled['Label'].values

        zd_label = np.unique(zero_day_sampled['Label']).tolist()
        y_test = np.where(y_test == zd_label, 1, 0)

        zd_df = pd.DataFrame(X_test, columns=zero_day_sampled.drop(columns=['Label']).columns)
        zd_df['Label'] = y_test
        benign_test['Label'] = np.where(benign_test['Label'] == 'Benign', 0, 1)

        client_test[i] = pd.concat([zd_df, benign_test], axis=0)
        client_test[i] = client_test[i].reset_index(drop=True)

        # scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(client_train[i].drop(columns=['Label']).values)
        X_test = scaler.transform(client_test[i].drop(columns=['Label']).values)

        # create the tensor dataset
        client_train[i] = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(2), torch.tensor(client_train[i]['Label'].values, dtype=torch.float32).unsqueeze(1))
        client_test[i] = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(2), torch.tensor(client_test[i]['Label'].values, dtype=torch.float32).unsqueeze(1))

        print('Finished label {}'.format(label))

        data_path = f"mydata/CICDDoS2019/zero_day_{which_label}/client_{i}/"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        torch.save(client_train[i], os.path.join(data_path, "train_data.pt"))
        torch.save(client_test[i], os.path.join(data_path, "test_data.pt"))

    else:
        break
        
