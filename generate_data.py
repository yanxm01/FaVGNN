import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
import random
import pickle
from scipy.spatial import distance_matrix

def load_data(dataset, sens_attr, predict_attr, path="../dataset/pokec/", seed=1225,
               test_idx=False, m=3, sens_rate=0.5):
    """Load data and split to m clients"""
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)

    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=np.int64)

    filtered_edges = []
    for edge in edges_unordered:
        if edge[0] in idx_map and edge[1] in idx_map:
            filtered_edges.append(edge)

    edges = np.array(list(map(idx_map.get, np.array(filtered_edges).flatten())), dtype=int).reshape(
        np.array(filtered_edges).shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    client_features = torch.chunk(features, m, dim=1)
    np.random.seed(seed)
    edge_indices = np.arange(edges.shape[0])
    np.random.shuffle(edge_indices)
    client_edges = np.array_split(edge_indices, m)
    clients_data = []
    for i in range(m):
        client_adj = sp.coo_matrix((np.ones(len(client_edges[i])),
                                    (edges[client_edges[i], 0], edges[client_edges[i], 1])),
                                   shape=(labels.shape[0], labels.shape[0]),
                                   dtype=np.float32)
        client_adj = client_adj + client_adj.T.multiply(client_adj.T > client_adj) - client_adj.multiply(
            client_adj.T > client_adj)
        client_adj = client_adj + sp.eye(client_adj.shape[0])
        clients_data.append((client_features[i], client_adj))


    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    idx_train = label_idx[:int(0.5 * len(label_idx))]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[int(0.75 * len(label_idx)):]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]
    sens = idx_features_labels[sens_attr].values
    sens = torch.FloatTensor(sens)
    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    client_data = []
    for i in range(m):
        sens_idx_client = set(np.where(sens >= 0)[0])
        idx_sens_train = list(sens_idx_client - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        num_sens_train = int(sens_rate * len(idx_train))
        idx_sens_train = torch.LongTensor(idx_sens_train[:num_sens_train])
        client_data.append({
            'features': clients_data[i][0],
            'adj': clients_data[i][1],
            'train_idx': torch.LongTensor(idx_train),
            'val_idx': torch.LongTensor(idx_val),
            'test_idx': torch.LongTensor(idx_test),
            'idx_sens_train': idx_sens_train
        })

    return client_data, labels, sens

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)

    return idx_map

def load_german_data(dataset, sens_attr, predict_attr, path="../dataset/german/", seed=1225,
               m=3, sens_rate=0.5):
    """Load data and split to m clients"""
    print('Loading {} dataset from {}'.format(dataset, path))
    # Sensitive Attribute
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    idx_features_labels.loc[idx_features_labels['Gender'] == 'Female', 'Gender'] = 0
    idx_features_labels.loc[idx_features_labels['Gender'] == 'Male', 'Gender'] = 1

    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')
    header.remove('Gender')

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    # build graph
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    client_features = torch.chunk(features, m, dim=1)
    np.random.seed(seed)
    edge_indices = np.arange(edges.shape[0])
    np.random.shuffle(edge_indices)
    client_edges = np.array_split(edge_indices, m)
    clients_data = []
    for i in range(m):
        client_adj = sp.coo_matrix((np.ones(len(client_edges[i])),
                                    (edges[client_edges[i], 0], edges[client_edges[i], 1])),
                                   shape=(labels.shape[0], labels.shape[0]),
                                   dtype=np.float32)
        client_adj = client_adj + client_adj.T.multiply(client_adj.T > client_adj) - client_adj.multiply(
            client_adj.T > client_adj)
        client_adj = client_adj + sp.eye(client_adj.shape[0])

        clients_data.append((client_features[i], client_adj))
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:int(0.5 * len(label_idx))]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    client_data = []
    for i in range(m):
        sens_idx_client = set(np.where(sens >= 0)[0])
        idx_sens_train = list(sens_idx_client - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        num_sens_train = int(sens_rate * len(idx_train))
        idx_sens_train = torch.LongTensor(idx_sens_train[:num_sens_train])


        client_data.append({
            'features': clients_data[i][0],
            'adj': clients_data[i][1],
            'train_idx': torch.LongTensor(idx_train),
            'val_idx': torch.LongTensor(idx_val),
            'test_idx': torch.LongTensor(idx_test),
            'idx_sens_train': idx_sens_train
        })

    return client_data, labels, sens

def save_data(client_data, labels, sens, dataset_name, sens_rate, save_path="./my_data/"):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    # Create filenames based on dataset_name and sens_rate
    filename_prefix = f"{dataset_name}_sens_{sens_rate}"
    with open(os.path.join(save_path, f"{filename_prefix}_client_data.pkl"), "wb") as f:
        pickle.dump(client_data, f)
    with open(os.path.join(save_path, f"{filename_prefix}_labels.pkl"), "wb") as f:
        pickle.dump(labels, f)
    with open(os.path.join(save_path, f"{filename_prefix}_sens.pkl"), "wb") as f:
        pickle.dump(sens, f)

def load_saved_data(dataset_name, sens_rate, load_path="./my_data/"):
    # Create filenames based on dataset_name and sens_rate
    filename_prefix = f"{dataset_name}_sens_{sens_rate}"

    with open(os.path.join(load_path, f"{filename_prefix}_client_data.pkl"), "rb") as f:
        client_data = pickle.load(f)

    with open(os.path.join(load_path, f"{filename_prefix}_labels.pkl"), "rb") as f:
        labels = pickle.load(f)

    with open(os.path.join(load_path, f"{filename_prefix}_sens.pkl"), "rb") as f:
        sens = pickle.load(f)

    return client_data, labels, sens

