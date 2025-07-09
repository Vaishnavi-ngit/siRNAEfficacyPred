import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, GraphConv
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy.stats
import json
import optuna

torch.cuda.manual_seed_all(42)

import BugUtils as utils1

with open("siRNA_param_pytorch.json", 'r') as f:
    base_params = json.load(f)

MAX_SIRNA_LENGTH = base_params["sirna_length"]

class HeteroSAGE(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0]

        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else -1
            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): GraphConv((in_channels, in_channels), out_channels_i),
                ('mRNA', 'interacts_with', 'interaction'): GraphConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'siRNA'): GraphConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'mRNA'): GraphConv((in_channels, in_channels), out_channels_i),
            }, aggr='mean')
            self.convs.append(conv)

            self.bns.append(nn.ModuleDict({
                node_type: BatchNorm1d(out_channels_i) for node_type in self.node_types
            }))

            self.dropouts.append(Dropout(dropout_rate))

        self.lin = Linear(layer_sizes[-1], out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict = conv(x_dict, edge_index_dict)

            for node_type in x_dict:
                if node_type in bn_dict:
                    if x_dict[node_type].size(0) > 1 or not self.training:
                        x_dict[node_type] = bn_dict[node_type](x_dict[node_type])
                x_dict[node_type] = F.leaky_relu(x_dict[node_type])
                x_dict[node_type] = dropout(x_dict[node_type])
        return self.lin(x_dict['interaction'])

    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())

print(f"Loaded Base Parameters: {base_params}")
print(f"Max siRNA Length for Padding: {MAX_SIRNA_LENGTH}")

TUNING_FOLD_IDX = 0

split_dir_tuning = f"siRNA_split_datasets/split{TUNING_FOLD_IDX}/"
if not os.path.exists(split_dir_tuning):
    raise FileNotFoundError(f"Error: Directory '{split_dir_tuning}' not found. Please ensure your data splits are correctly organized.")

try:
    data_train_tuning = pd.read_csv(os.path.join(split_dir_tuning, "train.csv"))
    data_dev_tuning = pd.read_csv(os.path.join(split_dir_tuning, "dev.csv"))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Error: Data file not found in '{split_dir_tuning}'. {e}")

data_train_tuning['split'] = 'train'
data_dev_tuning['split'] = 'dev'

for df in [data_train_tuning, data_dev_tuning]:
    df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')

data_tuning = pd.concat([data_train_tuning, data_dev_tuning], axis=0).reset_index(drop=True)

sirna_onehot = []
for seq in data_tuning['siRNA_seq']:
    sirna_onehot.append(utils1.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data_tuning['siRNA']))

mrna_onehot_temp = data_tuning.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
mrna_onehot = [utils1.obtain_one_hot_feature_for_one_sequence_1(seq, base_params["max_mrna_len"])
               for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))

sirna_pos_encoding = []
for idx, row in data_tuning.iterrows():
    mrna_start_pos = max(0, int(row['pos']))
    sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
        mrna_start_pos,
        len(row['siRNA_seq']),
        MAX_SIRNA_LENGTH,
        base_params["dmodel"]
    ))
sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data_tuning['siRNA'] + '_' + data_tuning['mRNA'])

sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                     for seq in data_tuning['siRNA_seq']]
sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
temp_interaction_index = data_tuning['siRNA'] + '_' + data_tuning['mRNA']
sirna_thermo_feat['index'] = temp_interaction_index
sirna_thermo_feat = sirna_thermo_feat.set_index('index')

con_feat = pd.read_csv("siRNA_split_preprocess/full_con_matrix.txt", header=None, index_col=0)
con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)

sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)

mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)

sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data_tuning['siRNA_seq']], index=list(data_tuning['siRNA']))
mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))

sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data_tuning['siRNA_seq']])
sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data_tuning['siRNA_seq']])
sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_tuning['siRNA_seq']])
sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_tuning['siRNA_seq']])
sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_tuning['siRNA_seq']])
sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
sirna_k_mers.index = data_tuning['siRNA']

SIRNA_RULES_LENGTH = 19
sirna_pos_scores = []
for seq in data_tuning['siRNA_seq']:
    sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data_tuning['siRNA']))

sirna_pd = pd.concat([sirna_onehot, sirna_sfold_feat, sirna_GC, sirna_k_mers, sirna_pos_scores], axis=1)
mrna_pd = pd.concat([mrna_onehot, mrna_sfold_feat, mrna_GC], axis=1)
interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding], axis=1)

data_hetero_tuning = HeteroData()
data_hetero_tuning['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
data_hetero_tuning['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
data_hetero_tuning['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)

sirna_name_to_idx = {name: i for i, name in enumerate(sirna_pd.index)}
mrna_name_to_idx = {name: i for i, name in enumerate(mrna_pd.index)}
interaction_name_to_idx = {name: i for i, name in enumerate(interaction_pd.index)}

edge_index_siRNA_to_interaction = []
edge_index_mRNA_to_interaction = []

for _, row in data_tuning.iterrows():
    siRNA_name = row['siRNA']
    mRNA_name = row['mRNA']
    interaction_name = f"{siRNA_name}_{mRNA_name}"

    if (siRNA_name in sirna_name_to_idx and
        mRNA_name in mrna_name_to_idx and
        interaction_name in interaction_name_to_idx):

        edge_index_siRNA_to_interaction.append([sirna_name_to_idx[siRNA_name], interaction_name_to_idx[interaction_name]])
        edge_index_mRNA_to_interaction.append([mrna_name_to_idx[mRNA_name], interaction_name_to_idx[interaction_name]])

data_hetero_tuning['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
    edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
data_hetero_tuning['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
    edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

data_hetero_tuning['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero_tuning['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
data_hetero_tuning['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero_tuning['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

train_idx_list = []
dev_idx_list = []

for _, row in data_train_tuning.iterrows():
    interaction_name = f"{row['siRNA']}_{row['mRNA']}"
    if interaction_name in interaction_name_to_idx:
        train_idx_list.append(interaction_name_to_idx[interaction_name])

for _, row in data_dev_tuning.iterrows():
    interaction_name = f"{row['siRNA']}_{row['mRNA']}"
    if interaction_name in interaction_name_to_idx:
        dev_idx_list.append(interaction_name_to_idx[interaction_name])

train_idx_tuning = torch.tensor(train_idx_list, dtype=torch.long)
dev_idx_tuning = torch.tensor(dev_idx_list, dtype=torch.long)

labels_tuning = torch.zeros(data_hetero_tuning['interaction'].num_nodes, dtype=torch.float)
for _, row in data_tuning.iterrows():
    interaction_name = f"{row['siRNA']}_{row['mRNA']}"
    if interaction_name in interaction_name_to_idx:
        labels_tuning[interaction_name_to_idx[interaction_name]] = row['efficacy']
data_hetero_tuning['interaction'].y = labels_tuning

def objective(trial: optuna.Trial):
    n_layers = trial.suggest_int('n_layers', 2, 4)

    initial_hidden_dim = trial.suggest_categorical('initial_hidden_dim', [256, 512, 1024])

    layer_sizes = [initial_hidden_dim]

    for i in range(1, n_layers):
        reduction_factor = trial.suggest_categorical(f'reduction_factor_layer_{i}', [2, 4, 8])
        next_dim = max(1, layer_sizes[-1] // reduction_factor)
        layer_sizes.append(next_dim)
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
    
    hop_samples = []
    for i in range(n_layers):
        hop_samples.append(trial.suggest_categorical(f'hop_sample_{i}', [5, 10, 20, 30]))
    
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    l1_lambda = trial.suggest_loguniform('l1_lambda', 1e-6, 1e-3)
    loss_function_choice = trial.suggest_categorical('loss_function', ["mse", "l1"])

    n_epochs = base_params["epochs"]

    train_loader = NeighborLoader(
        data_hetero_tuning,
        num_neighbors=hop_samples,
        batch_size=batch_size,
        input_nodes=('interaction', train_idx_tuning),
        shuffle=True,
        subgraph_type='induced',
        filter_per_worker=False
    )

    val_loader = NeighborLoader(
        data_hetero_tuning,
        num_neighbors=hop_samples,
        batch_size=batch_size,
        input_nodes=('interaction', dev_idx_tuning),
        shuffle=False,
        subgraph_type='induced',
        filter_per_worker=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroSAGE(
        layer_sizes=layer_sizes,
        out_channels=1,
        metadata=data_hetero_tuning.metadata(),
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    criterion = nn.MSELoss() if loss_function_choice == "mse" else nn.L1Loss()

    def train_epoch(loader, current_l1_lambda):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = criterion(out.squeeze(-1), batch['interaction'].y)
            loss += current_l1_lambda * model.l1_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def validate_epoch(loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            preds.append(out.cpu())
            truths.append(batch['interaction'].y.cpu())
        return torch.cat(preds), torch.cat(truths)

    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader, l1_lambda)
        val_pred, val_true = validate_epoch(val_loader)
        val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()

        scheduler.step(val_loss)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

if __name__ == '__main__':
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        ),
        storage='sqlite:///siRNA_gnn_optuna.db',
        study_name='siRNA_GNN_Hyperparameter_Tuning',
        load_if_exists=True # Added this line to load the study if it exists
    )

    study.optimize(objective, n_trials=50, gc_after_trial=True)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params_path = "best_siRNA_gnn_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)