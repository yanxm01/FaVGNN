import torch
from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GINConv, SAGEConv
from torch.nn.utils import spectral_norm
import torch.optim as optim

# class MLP(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def reset_parameters(self):
#         self.fc1.reset_parameters()
#         self.fc2.reset_parameters()
#         self.fc3.reset_parameters()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,  lr, weight_decay):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_mlp_state = None

    def forward(self, x):
        return self.model(x)

    def reset_parameters(self):
        for layer in self.model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train_model(self, X_train, y_train, X_val, y_val, epochs, logger):

        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(X_train)
            train_loss = nn.functional.mse_loss(output, y_train)
            train_loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output = self.model(X_val)
                val_loss = nn.functional.mse_loss(output, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_mlp_state = self.model.state_dict()

            if epoch % 5 == 0:  # 每5个epoch输出一次日志
                logger.info(f'Epoch: {epoch} \tTraining Loss: {train_loss:.4f}\tValidation Loss: {val_loss:.4f}')

        self.model.load_state_dict(self.best_mlp_state)

class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)

        return torch.sigmoid(h)


class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h


class GCN_encoder_scatter(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = propagate2(h, edge_index) + self.bias
        return h


class GCN_encoder_spmm(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_spmm, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)
        self.bias = Parameter(torch.Tensor(args.hidden))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = torch.spmm(adj_norm_sp, h) + self.bias

        return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h = x
        # h = self.conv2(x, edge_index)
        return h


class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h


