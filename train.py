import os
import pickle
import gc
import sys
import argparse
import numpy as np

import torch 
from utils import *

from minisom import MiniSom
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from som_dagmm.model import DAGMM, SOM_DAGMM
from som_dagmm.compression_network import CompressionNetwork
from som_dagmm.estimation_network import EstimationNetwork
from som_dagmm.gmm import GMM, Mixture

from SOM import som_train, som_pred



# Aqui é somente pegar as informações da linha de comando
def parse_args():
    
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vehicle_claims', type=str)
    parser.add_argument('--embedding', dest='embed', help='one_hot, label', default='NULL', type=str)
    parser.add_argument('--features', dest='features', help='all, numerical, categorical', default='all', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='32', default = 32, type=int)
    parser.add_argument('--epoch', dest='epoch', help='1', default='1', type=int)
    args = parser.parse_args()
    return args

args = parse_args()
epochs = args.epoch
batch_size = args.batch_size
save_path = os.path.join(args.dataset + "_" + args.features + "_" + args.embed)
#read data
# get labels from dataset and drop them if available

data = pd.DataFrame()

if args.dataset == 'IDS2018':
    data_list = []  # Lista para armazenar os DataFrames a serem concatenados
    for d in os.listdir('data/CSE-CIC-IDS2018'):
        if d != '.ipynb_checkpoints':
            new_data = load_data(f'data/CSE-CIC-IDS2018/{d}')
            data_list.append(new_data)  # Adiciona o novo DataFrame à lista
    
    # Concatena todos os DataFrames na lista
    data = pd.concat(data_list, ignore_index=True)
    data.drop(['  q   q   q   Timestamp', 'Timestamp'], axis=1, inplace=True)
    categorical_cols = []
    Y = get_labels(data, args.dataset)

if args.dataset == 'arrhythmia':
    data = load_data('data/arrhythmia.csv')
    data = remove_cols(data, ['J'])
    Y = get_labels(data, args.dataset)
if args.dataset == 'kdd':
    names = [i for i in range(0,43)] # Qtd de colunas, cada coluna está representada entre 0 a 42
    data = load_data('data/NSL-KDD/KDDTrain+.txt', names)

    data = data[(data[41] ==  "normal")]

    categorical_cols = [1,2,3] # Somente as colunas 1,2,3
    Y = get_labels(data, args.dataset)

#Select features
if args.features == "categorical":
    data = data[categorical_cols]
if args.features == "numerical":
    data = remove_cols(data, categorical_cols)

#encode categorical variables 
if args.embed == 'one_hot':
    data = one_hot_encoding(data, categorical_cols)
if args.embed == 'label_encode':
    data = label_encoding(data, categorical_cols)

# Remove columns with NA values
data = fill_na(data)
# normalize data
data = normalize_cols(data)
#test and train split
train_data, test_data, Y_train, Y_test = split_data(data, Y, 0.8)

#train_data = train_data.values.astype(np.float32)
print(train_data.shape)

#Convert to torch tensors
data = torch.tensor(data.values.astype(np.float32))
train_data = torch.tensor(train_data.values.astype(np.float32))
test_data = torch.tensor(test_data.values.astype(np.float32))

#Convert tensor to TensorDataset class.
dataset = TensorDataset(data)

#TrainLoader
dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True)



compression = CompressionNetwork(data.shape[1])
estimation = EstimationNetwork()
gmm = GMM(2,6)
mix = Mixture(6)
dagmm = DAGMM(compression, estimation, gmm)
net = SOM_DAGMM(dagmm)
optimizer =  optim.Adam(net.parameters(), lr=1e-4)
for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch + 1))
    running_loss = 0
    for i, data in enumerate(dataloader):
        out = net(data[0])
        optimizer.zero_grad()
        L_loss = compression.reconstruction_loss(data[0])
        G_loss = mix.gmm_loss(out=out, L1=0.1, L2=0.005)
        loss = L_loss + G_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss)
torch.save(net, save_path)
