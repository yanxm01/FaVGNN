import argparse
import numpy as np
import torch.nn as nn
import torch
from utils import accuracy
from models.FaVGNN import FaVGNN
import dgl
from utils import feature_norm
from sklearn.metrics import roc_auc_score
from train_estimator import train_estimator
import logging
import os
from datetime import datetime
import json
from models.MLP import MLP
import pickle
from sklearn.model_selection import train_test_split

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def average_classifiers(classifiers):
    global_classifier_state = classifiers[0].state_dict()
    for key in global_classifier_state:
        global_classifier_state[key] = torch.mean(
            torch.stack([classifier.state_dict()[key] for classifier in classifiers]), dim=0)
    return global_classifier_state

def load_saved_data(dataset_name, sens_rate, load_path="./my_data/"):
    filename_prefix = f"{dataset_name}_sens_{sens_rate}"
    with open(os.path.join(load_path, f"{filename_prefix}_client_data.pkl"), "rb") as f:
        client_data = pickle.load(f)
    with open(os.path.join(load_path, f"{filename_prefix}_labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    with open(os.path.join(load_path, f"{filename_prefix}_sens.pkl"), "rb") as f:
        sens = pickle.load(f)
    return client_data, labels, sens

def get_neighbor(client_data, sens_features, idx_train):
    idx_train = idx_train.cpu()
    client_adj = torch.zeros((client_data['adj'].shape[0], client_data['adj'].shape[0])).int()
    for j in range(client_data['adj'].shape[0]):
        nonzero_elements = np.array(client_data['adj'][j].nonzero())
        neighbor = torch.tensor(nonzero_elements)
        mask = (sens_features[neighbor[1]] != sens_features[j])
        h_nei_idx = neighbor[1][mask]
        client_adj[j, h_nei_idx] = 1
    train_adj = client_adj[idx_train][:, idx_train]
    features = client_data['features'].cpu()
    c_X = torch.cat((features[idx_train], sens_features[idx_train].unsqueeze(1)), dim=1)
    train_adj = train_adj.cpu()
    deg = np.sum(train_adj.numpy(), axis=1)
    deg[deg == 0] = 1
    deg = torch.from_numpy(deg).cpu()
    indices = torch.nonzero(train_adj)
    values = train_adj[indices[:, 0], indices[:, 1]]
    mat = torch.sparse_coo_tensor(indices.t(), values, train_adj.shape).float().cpu()
    h_X = torch.spmm(mat, c_X.cpu()) / deg.unsqueeze(-1)
    mask = torch.any(torch.isnan(h_X), dim=1)
    h_X = h_X[~mask]
    c_X = c_X[~mask]
    indices = np.arange(c_X.shape[0])
    [indices_train, indices_test, y_train, y_test] = train_test_split(indices, indices, test_size=0.1)
    X_train, X_test, y_train, y_test = c_X[indices_train], c_X[indices_test], h_X[indices_train], h_X[indices_test]
    del client_adj
    del neighbor
    client_neighbor_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return client_neighbor_data

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='german',
                    choices=['pokec_z', 'pokec_n', 'german'])
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1225, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units of the sensitive attribute estimator')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--alpha', type=float, default=5,
                    help='The hyperparameter of alpha')
parser.add_argument('--beta', type=float, default=0.5,
                    help='The hyperparameter of beta')
parser.add_argument('--theta', type=float, default=0.1,
                    help='The hyperparameter of theta')

parser.add_argument('--model', type=str, default="GAT",
                    help='the type of model GCN/GAT')
parser.add_argument('--num-hidden', type=int, default=64,
                    help='Number of hidden units of classifier.')
parser.add_argument("--num-heads", type=int, default=1,
                    help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--attn-drop", type=float, default=.0,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--clients', type=int, default=3,
                    help="number of clients")
parser.add_argument('--sens_rate', type=float, default=0.1,
                    help="proportion of sensitive attributes in each client's training set")

parser.add_argument('--pre_epoch', type=int, default=5,
                    help='Number of epochs to train in first communication.')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train.')
parser.add_argument('--e_epochs', type=int, default=200,
                    help='Number of epochs to pretrain estimator.')
parser.add_argument('--m_lr', type=float,  default=0.1,help='lr for mlp optimizer')
parser.add_argument('--hidden_mlp', type=int, default=16,
                    help='Number of hidden units of the mlp')
parser.add_argument('--m_epoch', type=int, default=20,
                    help='Number of epochs to train mlp.')
parser.add_argument('--f_epoch', type=int, default=2000,
                    help='Number of epochs to train fairgnn.')
parser.add_argument('--delta', type=float, default=1)

parser.add_argument('--g_steps', type=int, default=100,
                    help='Number of epochs to train fairgnn.')
parser.add_argument('--a_steps', type=int, default=50,
                    help='Number of epochs to train fairgnn.')

parser.add_argument('--com_epoch', type=int, default=10,
                    help='communication epoch')

parser.add_argument('--temp', type=float, default=0.02,
                    help="a scaling factor used to control the smoothness of the similarity")

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
# %%
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def fair_metric(output, idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)
    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))
    return parity, equality

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'log_train'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
folder_name = f'{args.dataset}_sens_{args.sens_rate}_{current_time}'
log_folder_path = os.path.join(log_dir, folder_name)
if not os.path.exists(log_folder_path):
    os.makedirs(log_folder_path)
log_file_name = f'{folder_name}.log'
log_file_path = os.path.join(log_folder_path, log_file_name)
logger = logging.getLogger(f'logger_{args.sens_rate}')
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console_handler)
args_file_path = os.path.join(log_folder_path, 'args.json')
with open(args_file_path, 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)

# load dataset
client_data, labels, sens = load_saved_data(args.dataset, args.sens_rate)
for i, client in enumerate(client_data):
    client['graph'] = dgl.from_scipy(client['adj'])
    client['graph'] = client['graph'].to('cuda:0')
    if args.dataset == 'nba':
        client['features'] = feature_norm(client['features'])
labels[labels > 1] = 1
sens[sens > 0] = 1

# build model
client_model = []
client_mlp = []
for m in range(args.clients):
    model = FaVGNN(nfeat=client_data[m]['features'].shape[1], args=args)
    mlp_model = MLP(input_dim=client_data[m]['features'].shape[1]+1, hidden_dim=args.hidden_mlp, output_dim=client_data[m]['features'].shape[1]+1,
                    lr=args.m_lr, weight_decay=0.001)
    if args.cuda:
        model.cuda()
        mlp_model = mlp_model.cuda()
        client_data[m]['features'] = client_data[m]['features'].cuda()
        labels = labels.cuda()
        client_data[m]['train_idx'] = client_data[m]['train_idx'].cuda()
        client_data[m]['val_idx'] = client_data[m]['val_idx'].cuda()
        client_data[m]['test_idx'] = client_data[m]['test_idx'].cuda()
        sens = sens.cuda()
        client_data[m]['idx_sens_train'] = client_data[m]['idx_sens_train'].cuda()
    client_model.append(model)
    client_mlp.append(mlp_model)
global_classifier = nn.Linear(args.num_hidden, 1)
if args.cuda:
    global_classifier.cuda()
global_classifier_state = global_classifier.state_dict()
for model in client_model:
    model.classifier.load_state_dict(global_classifier_state)

# pretrain estimator
for m in range(args.clients):
    model = client_model[m]
    logger.info(f'pretrain client {m} estimator ')
    train_estimator(model, client_data[m]['graph'], client_data[m]['features'], sens,
                    client_data[m]['idx_sens_train'], args, logger)

client_features = []
client_sims = []
client_acc_metrics = np.array([])
client_fair_metrics = np.array([])
client_metrics = {m: {'acc': [], 'auc': [], 'sp': [], 'eo': []} for m in range(args.clients)}
server_metrics = {
    'acc_test': [],
    'roc_test': [],
    'parity': [],
    'equality': []
}

# train model
for com_epoch in range(args.com_epoch):
    client_features = []
    client_sims = []
    client_acc_metrics = np.array([])
    client_fair_metrics = np.array([])

    # client train
    for m in range(args.clients):
        client_model[m].classifier.load_state_dict(global_classifier_state)
        logger.info(f"================client {m} train in com_epoch {com_epoch + 1}=================")
        if com_epoch == 0:
            for pre_epoch in range(args.pre_epoch):
                logger.info(f"=============client {m} train in com_epoch {com_epoch + 1}  in epoch {pre_epoch}===============")
                client_model[m].eval()
                with torch.no_grad():
                    predict_sens = client_model[m].estimator(client_data[m]['graph'], client_data[m]['features'])
                predict_sens_new = predict_sens.detach().cpu().squeeze()
                predict_sens_new[client_data[m]['idx_sens_train']] = sens[client_data[m]['idx_sens_train']].cpu()
                client_neighbor_data = get_neighbor(client_data[m], predict_sens_new, client_data[m]['train_idx'])
                del predict_sens_new
                logger.info(f'============client {m} mlp train in com_epoch {com_epoch + 1}  in epoch {pre_epoch}===============')
                client_mlp[m].train_model(client_neighbor_data['X_train'].cuda(),
                                          client_neighbor_data['y_train'].cuda(),
                                          client_neighbor_data['X_test'].cuda(),
                                          client_neighbor_data['y_test'].cuda(), args.m_epoch, logger)
                client_mlp[m].eval()

                mlp_save_path = os.path.join(log_folder_path, f'client{m}_com-epoch{com_epoch + 1}_epoch{pre_epoch}_mlp.pth')
                torch.save(client_mlp[m].state_dict(), mlp_save_path)

                del client_neighbor_data
                torch.cuda.empty_cache()

                all_features = torch.cat((client_data[m]['features'], predict_sens.detach()), dim=1)
                with torch.no_grad():
                    neighbor_features = client_mlp[m](all_features)
                fused_features = all_features + args.delta * neighbor_features
                logger.info(f'============client {m} fairgnn train  in com_epoch {com_epoch + 1} in epoch {pre_epoch}===============')
                model = client_model[m]
                for f_epoch in range(args.f_epoch):
                    model.train()
                    model.optimize(client_data[m]['graph'], client_data[m]['features'], fused_features, labels,
                                   client_data[m]['train_idx'], sens, client_data[m]['idx_sens_train'])
                    cov = model.cov
                    cls_loss = model.cls_loss
                    adv_loss = model.adv_loss

                model_save_path = os.path.join(log_folder_path,f'client{m}_com-epoch{com_epoch + 1}_epoch{pre_epoch}_model.pth')
                torch.save(model.state_dict(), model_save_path)

                model.eval()
                with torch.no_grad():
                    output, s = model(client_data[m]['graph'], client_data[m]['features'], fused_features)
                idx_val = client_data[m]['val_idx']
                acc_val = accuracy(output[idx_val], labels[idx_val])
                roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
                idx_test = client_data[m]['test_idx']
                acc_sens = accuracy(s[idx_test], sens[idx_test])
                parity_val, equality_val = fair_metric(output, idx_val)
                acc_test = accuracy(output[idx_test], labels[idx_test])
                roc_test = roc_auc_score(labels[idx_test].cpu().numpy(),
                                         output[idx_test].detach().cpu().numpy())
                parity, equality = fair_metric(output, idx_test)
                logger.info(f'Epoch: {f_epoch + 1:04d} '
                            f'cov: {cov.item():.4f} '
                            f'cls: {cls_loss.item():.4f} '
                            f'adv: {adv_loss.item():.4f} '
                            f'acc_val: {acc_val.item():.4f} '
                            f'roc_val: {roc_val:.4f} '
                            f'parity_val: {parity_val:.4f}'
                            f'equality_val: {equality_val:.4f}')
                logger.info(f'Test: accuracy: {acc_test.item():.4f} '
                            f'roc: {roc_test:.4f} '
                            f'acc_sens: {acc_sens:.4f} '
                            f'parity: {parity:.4f} '
                            f'equality: {equality:.4f}')
            logger.info(f"================client {m} train finished!=================")

        else:
            for epoch in range(args.epochs):
                logger.info(f"=============client {m} train in com_epoch {com_epoch + 1} in epoch {epoch}===============")
                client_model[m].eval()
                with torch.no_grad():
                    predict_sens = client_model[m].estimator(client_data[m]['graph'], client_data[m]['features'])
                predict_sens_new = predict_sens.detach().cpu().squeeze()
                predict_sens_new[client_data[m]['idx_sens_train']] = sens[client_data[m]['idx_sens_train']].cpu()

                client_neighbor_data = get_neighbor(client_data[m], predict_sens_new, client_data[m]['train_idx'])
                del predict_sens_new
                torch.cuda.empty_cache()

                logger.info(f'============client {m} mlp train in com_epoch {com_epoch + 1} in epoch {epoch}==============')
                client_mlp[m].train_model(client_neighbor_data['X_train'].cuda(), client_neighbor_data['y_train'].cuda(), client_neighbor_data['X_test'].cuda(),client_neighbor_data['y_test'].cuda(), args.m_epoch, logger)
                client_mlp[m].eval()

                mlp_save_path = os.path.join(log_folder_path, f'client{m}_com-epoch{com_epoch + 1}_epoch{epoch}_mlp.pth')
                torch.save(client_mlp[m].state_dict(), mlp_save_path)

                del client_neighbor_data
                torch.cuda.empty_cache()

                all_features = torch.cat((client_data[m]['features'], predict_sens.detach()), dim=1)
                with torch.no_grad():
                    neighbor_features = client_mlp[m](all_features)
                fused_features = all_features + args.delta * neighbor_features

                del all_features, neighbor_features, predict_sens
                torch.cuda.empty_cache()

                logger.info(f'============client {m} fairgnn train in com_epoch {com_epoch + 1} in epoch {epoch}===============')
                model = client_model[m]
                for f_epoch in range(args.f_epoch):
                    model.train()
                    model.col_optimize(client_data[m]['graph'], client_data[m]['features'], fused_features, labels,
                                   client_data[m]['train_idx'], sens, client_data[m]['idx_sens_train'], server_sim, args.temp)
                    sim_loss = model.sim_loss
                    cov = model.cov
                    cls_loss = model.cls_loss
                    adv_loss = model.adv_loss

                    if (f_epoch + 1) % 1000 == 0 or f_epoch == args.f_epoch-1:
                        model.eval()
                        with torch.no_grad():
                            output, s = model(client_data[m]['graph'], client_data[m]['features'], fused_features)
                        idx_val = client_data[m]['val_idx']
                        acc_val = accuracy(output[idx_val], labels[idx_val])
                        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
                        idx_test = client_data[m]['test_idx']
                        acc_sens = accuracy(s[idx_test], sens[idx_test])
                        parity_val, equality_val = fair_metric(output, idx_val)
                        acc_test = accuracy(output[idx_test], labels[idx_test])
                        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
                        parity, equality = fair_metric(output, idx_test)

                        logger.info(f'Epoch: {f_epoch + 1:04d} '
                                    f'cov: {cov.item():.4f} '
                                    f'cls: {cls_loss.item():.4f} '
                                    f'adv: {adv_loss.item():.4f} '
                                    f'sim: {sim_loss.item():.4f} '
                                    f'acc_val: {acc_val.item():.4f} '
                                    f'roc_val: {roc_val:.4f} '
                                    f'parity_val: {parity_val:.4f}'
                                    f'equality_val: {equality_val:.4f}')
                        logger.info(f'Test: accuracy: {acc_test.item():.4f} '
                                    f'roc: {roc_test:.4f} '
                                    f'acc_sens: {acc_sens:.4f} '
                                    f'parity: {parity:.4f} '
                                    f'equality: {equality:.4f}')
                        logger.info(f"================client {m} train finished!=================")
                model_save_path = os.path.join(log_folder_path,f'client{m}_com-epoch{com_epoch + 1}_epoch{epoch}_model.pth')
                torch.save(model.state_dict(), model_save_path)

        client_metrics[m]['acc'].append(acc_test.item())
        client_metrics[m]['auc'].append(roc_test)
        client_metrics[m]['sp'].append(parity)
        client_metrics[m]['eo'].append(equality)

        del output, s
        torch.cuda.empty_cache()

        client_feature = model.get_embeddings(client_data[m]['graph'], fused_features)
        del fused_features
        torch.cuda.empty_cache()

        client_features.append(client_feature)

        client_sim = model.calculate_sim(client_feature[client_data[m]['train_idx']], args.temp)
        client_sims.append(client_sim.clone().detach())

        fair_metrics = parity + equality
        acc_metrics = acc_test + roc_test
        client_fair_metrics = np.append(client_fair_metrics, fair_metrics)
        client_acc_metrics = np.append(client_acc_metrics, acc_metrics.cpu().numpy())

    # server
    logger.info(f'============com_epoch {com_epoch + 1} server aggregation =============')
    diff_metrics = client_acc_metrics - client_fair_metrics
    weights = np.exp(diff_metrics) / np.sum(np.exp(diff_metrics))
    weights = torch.tensor(weights, dtype=torch.float32)

    weighted_features = [weights[i] * client_features[i] for i in range(len(client_features))]
    global_features = sum(weighted_features)
    global_features = global_features.cuda()

    weighted_sims = [weights[i] * client_sims[i] for i in range(len(client_sims))]
    global_sims = sum(weighted_sims)
    server_sim = global_sims.cuda()

    client_classifiers = [client_model[i].classifier for i in range(args.clients)]
    global_classifier_state = average_classifiers(client_classifiers)
    global_classifier.load_state_dict(global_classifier_state)
    y_server = global_classifier(global_features)

    idx_val = client_data[0]['val_idx']
    acc_val = accuracy(y_server[idx_val], labels[idx_val])
    roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), y_server[idx_val].detach().cpu().numpy())
    idx_test = client_data[0]['test_idx']
    parity_val, equality_val = fair_metric(y_server, idx_val)
    acc_test = accuracy(y_server[idx_test], labels[idx_test])
    roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), y_server[idx_test].detach().cpu().numpy())
    parity, equality = fair_metric(y_server, idx_test)
    logger.info('============performance on server aggregation=============')
    logger.info(f'Epoch: {com_epoch + 1:04d} '
                f'acc_val: {acc_val.item():.4f} '
                f'roc_val: {roc_val:.4f} '
                f'parity_val: {parity_val:.4f}'
                f'equality_val: {equality_val:.4f}')
    logger.info(f'Test: accuracy: {acc_test.item():.4f} '
                f'roc: {roc_test:.4f} '
                f'parity: {parity:.4f} '
                f'equality: {equality:.4f}')

    server_metrics['acc_test'].append(acc_test.item())
    server_metrics['roc_test'].append(roc_test)
    server_metrics['parity'].append(parity)
    server_metrics['equality'].append(equality)


