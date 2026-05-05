import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv
import torch.nn.functional as F


# ===============================
# Stable variance activation
# ===============================
def positive_var(x, eps=1e-6):
    return F.softplus(x) + eps


# ===============================
# MC Dropout
# ===============================
class MCDropout(nn.Module):
    def __init__(self, p=0.05):   # smaller dropout
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


# ===============================
# Simple Probabilistic Linear
# ===============================
class ProbLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.logvar = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        mu = self.mu(x)
        var = positive_var(self.logvar(x))
        return mu, var


# ===============================
# MODEL
# ===============================
class BayesianPowerGridGNN(nn.Module):
    def __init__(self, var_in_dim, factor_in_dim, hidden_dim,
                 readout_units, output_units, num_iterations=3):

        super().__init__()
        self.num_iterations = num_iterations

        # encoders
        self.var_encoder = nn.Linear(var_in_dim, hidden_dim)
        self.factor_encoder = nn.Linear(factor_in_dim, hidden_dim)

        # message passing
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        self.update = nn.ModuleList()

        for _ in range(num_iterations):
            self.stage1.append(HeteroConv({
                ('factor', 'to', 'variable'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                ('variable', 'to', 'variable'): GATConv(-1, hidden_dim, add_self_loops=False),
            }, aggr='sum'))

            self.stage2.append(HeteroConv({
                ('variable', 'to', 'factor'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            }, aggr='sum'))

            self.update.append(nn.ModuleDict({
                'variable': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ),
                'factor': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
            }))

        # readout (mean + variance)
        self.readout_mu = nn.Sequential(
            nn.Linear(hidden_dim, readout_units),
            nn.ReLU(),
            nn.Linear(readout_units, output_units)
        )

        self.readout_var = nn.Sequential(
            nn.Linear(hidden_dim, readout_units),
            nn.ReLU(),
            nn.Linear(readout_units, output_units)
        )

        self.dropout = MCDropout(0.05)

    def forward(self, x_dict, edge_index_dict):

        x_dict = {
            'variable': self.var_encoder(x_dict['variable']),
            'factor': self.factor_encoder(x_dict['factor'])
        }

        for i in range(self.num_iterations):
            out1 = self.stage1[i](x_dict, edge_index_dict)
            x_dict['variable'] = self.update[i]['variable'](out1['variable']) + x_dict['variable']

            out2 = self.stage2[i](x_dict, edge_index_dict)
            x_dict['factor'] = self.update[i]['factor'](out2['factor']) + x_dict['factor']

        h = x_dict['variable']

        mu = self.readout_mu(h)
        var = positive_var(self.readout_var(self.dropout(h)))

        return {'mu': mu, 'var': var}


    @torch.no_grad()
    def predict_with_uncertainty(self, x_dict, edge_index_dict, T=30):
        preds = []

        for _ in range(T):
            out = self.forward(x_dict, edge_index_dict)
            preds.append(out['mu'])

        preds = torch.stack(preds)

        mean = preds.mean(0)
        epistemic = preds.var(0)

        out = self.forward(x_dict, edge_index_dict)
        aleatoric = out['var']

        return {
            'mu': mean,
            'var_epistemic': epistemic,
            'var_aleatoric': aleatoric,
            'var_total': epistemic + aleatoric
        }