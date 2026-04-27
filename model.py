import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv

class PowerGridGNN(torch.nn.Module):
    def __init__(self, var_in_dim, factor_in_dim, hidden_dim, readout_units, output_units, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations

        # ==========================================
        # 1. INITIAL STATE ENCODERS (build_state)
        # Maps your raw features into the 'hidden_state_dim'
        # ==========================================
        self.var_encoder = nn.Linear(var_in_dim, hidden_dim)
        self.factor_encoder = nn.Linear(factor_in_dim, hidden_dim)

        # ==========================================
        # 2. MESSAGE PASSING STAGES
        # We use ModuleLists because we repeat this 'num_iterations' times (N in YAML)
        # ==========================================
        self.stage1_convs = torch.nn.ModuleList()
        self.stage2_convs = torch.nn.ModuleList()
        self.update_nns = torch.nn.ModuleList()

        for _ in range(num_iterations):
            # STAGE 1: factorNode & variableNode -> variableNode
            conv1 = HeteroConv({
                # factorToVariable Message NN + Attention
                ('factor', 'to', 'variable'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
                # variableToVariable Message NN + Attention
                ('variable', 'to', 'variable'): GATConv(-1, hidden_dim, add_self_loops=False),
            }, aggr='sum')
            self.stage1_convs.append(conv1)

            # STAGE 2: variableNode -> factorNode
            conv2 = HeteroConv({
                # variableToFactor Message NN + Attention
                ('variable', 'to', 'factor'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            }, aggr='sum')
            self.stage2_convs.append(conv2)

            # UPDATE NNs (1 layer Dense with ReLU as per your YAML)
            update_dict = torch.nn.ModuleDict({
                'variable': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ),
                'factor': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
            })
            self.update_nns.append(update_dict)

        # ==========================================
        # 3. READOUT MODEL
        # Maps the final hidden state of variableNodes to the output voltage
        # ==========================================
        self.readout_model = nn.Sequential(
            nn.Linear(hidden_dim, readout_units),
            nn.ReLU(),
            nn.Linear(readout_units, output_units)
            # activation: None (No activation at the end, just raw linear output)
        )

    def forward(self, x_dict, edge_index_dict):
        # Step 1: Encode initial features into hidden dimensions
        x_dict = {
            'variable': self.var_encoder(x_dict['variable']),
            'factor': self.factor_encoder(x_dict['factor'])
        }

        # Step 2: Run the message passing loop N times
        for i in range(self.num_iterations):
            # --- STAGE 1: Update Variables ---
            out_stage1 = self.stage1_convs[i](x_dict, edge_index_dict)
            
            # Apply Update NN and a residual connection
            x_dict['variable'] = self.update_nns[i]['variable'](out_stage1['variable']) + x_dict['variable']

            # --- STAGE 2: Update Factors ---
            # Now use the *updated* variables to update the factors
            out_stage2 = self.stage2_convs[i](x_dict, edge_index_dict)
            
            # Apply Update NN and a residual connection
            x_dict['factor'] = self.update_nns[i]['factor'](out_stage2['factor']) + x_dict['factor']

        # Step 3: Readout Phase (Only care about variable nodes for final output)
        final_variable_states = x_dict['variable']
        output_voltage = self.readout_model(final_variable_states)

        # Return as a dictionary to match the training loop logic
        return {'variable': output_voltage}