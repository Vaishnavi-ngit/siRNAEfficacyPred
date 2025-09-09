import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch.nn import LayerNorm, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.transforms import ToUndirected
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import scipy.stats
import json
import math
import re
import E_Utils as utils

params = json.load(open("siRNA_param_pytorch.json", 'r'))

score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class HeteroSAGE(nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()
        self.node_types = metadata[0]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Project all node types to first hidden dimension
        self.input_proj = nn.ModuleDict({
            node_type: Linear(-1, layer_sizes[0]) for node_type in self.node_types
        })

        # Define HeteroConv layers with normalization and dropout
        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else layer_sizes[0]
            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('mRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'siRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'mRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
            }, aggr='mean')

            norm = nn.ModuleDict({
                node_type: nn.LayerNorm(out_channels_i) for node_type in self.node_types
            })

            self.convs.append(conv)
            self.norms.append(norm)
            self.dropouts.append(nn.Dropout(p=dropout_rate + 0.15))

        # Final MLP head (non-linear for better regression)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(layer_sizes[-1]),
            nn.GELU(),
            nn.Dropout(p=dropout_rate + 0.1),
            Linear(layer_sizes[-1], out_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.input_proj.values():
            lin.reset_parameters()
        for conv in self.convs:
            for layer in conv.convs.values():
                layer.reset_parameters()
        for layer in self.mlp_head:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        # Project input features to shared space
        x_dict = {
            node_type: self.input_proj[node_type](x)
            for node_type, x in x_dict.items()
        }

        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            h_dict = conv(x_dict, edge_index_dict)
            for node_type in h_dict:
                h = norm[node_type](h_dict[node_type])
                h = F.gelu(h)
                h = dropout(h)
                # Residual connection
                if h.shape == x_dict[node_type].shape:
                    h = h + x_dict[node_type]
                x_dict[node_type] = h

        # MLP head on 'interaction' node
        return self.mlp_head(x_dict['interaction'])

    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())


print(f"Dropout Rate: {params['dropout']}")
print(f"Learning Rate: {params['lr']}")
print(f"Epochs: {params['epochs']}")

for n in range(10):
    print(f"\n{'='*10} Processing Fold {n} {'='*10}")

    data_train = pd.read_csv(f"siRNA_split_datasets/split{n}/train.csv")
    data_dev = pd.read_csv(f"siRNA_split_datasets/split{n}/dev.csv")
    data_test = pd.read_csv(f"siRNA_split_datasets/split{n}/test.csv")

    data_train['split'] = 'train'
    data_dev['split'] = 'dev'
    data_test['split'] = 'test'

    for df in [data_train, data_dev, data_test]:
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('T', 'U', regex=False)
        df['mRNA_seq'] = df['mRNA_seq'].str.replace('T', 'U', regex=False)

    data = pd.concat([data_train, data_dev, data_test], axis=0)

    print("--- Starting Sequence Embedding (siRNA) ---")
    sirna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in data['siRNA_seq']]
    sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(data['siRNA']))
    print(f"----------------CHECKING---------------")
    sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')]
    print("--- Finished Sequence Embedding (siRNA) ---")

    print("--- Starting Sequence Embedding (mRNA) ---")
    mrna_unique_seq_df = data.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
    mrna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index=list(mrna_unique_seq_df['mRNA']))
    mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')]
    print("--- Finished Sequence Embedding (mRNA) ---")

    trans_table = str.maketrans('AUCG', 'UAGC')
    data['match_pos_siRNA_rev_comp'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
    data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].find(row['match_pos_siRNA_rev_comp']), axis=1)

    sirna_length = params.get("sirna_length", 19)
    dmodel = params.get("dmodel", 64)
    sirna_pos_encoding_per_interaction = [utils.get_pos_embedding_sequence(num, sirna_length, dmodel) for num in data['match_pos']]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding_per_interaction, index=data['siRNA'] + '_' + data['mRNA'])
    sirna_pos_encoding = sirna_pos_encoding.loc[~sirna_pos_encoding.index.duplicated(keep='first')]

    sirna_thermo_feat_raw = [utils.cal_thermo_feature(seq) for seq in data['siRNA_seq']]
    sirna_thermo_feat_df = pd.DataFrame(sirna_thermo_feat_raw)

    sirna_thermo_feat = pd.concat([data['siRNA'].reset_index(drop=True),
                                   data['mRNA'].reset_index(drop=True),
                                   sirna_thermo_feat_df],
                                  axis=1)
    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])
    sirna_thermo_feat = sirna_thermo_feat.loc[~sirna_thermo_feat.index.duplicated(keep='first')]

    con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)

    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0)

    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0)

    sirna_GC = [utils.countGC(seq) for seq in data['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC, columns=['GC_content'], index=list(data['siRNA']))
    sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')]

    mrna_GC = [utils.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, columns=['GC_content'], index=list(mrna_unique_seq_df['mRNA']))
    mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')]

    print("--- Starting AU Content Calculation ---")
    sirna_AU = [utils.countAU(seq) for seq in data['siRNA_seq']]
    sirna_AU = pd.DataFrame(sirna_AU, columns=['AU_content'], index=list(data['siRNA']))
    sirna_AU = sirna_AU.loc[~sirna_AU.index.duplicated(keep='first')]

    mrna_AU = [utils.countAU(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_AU = pd.DataFrame(mrna_AU, columns=['AU_content'], index=list(mrna_unique_seq_df['mRNA']))
    mrna_AU = mrna_AU.loc[~mrna_AU.index.duplicated(keep='first')]
    print("--- Finished AU Content Calculation ---")

    print("--- Starting Regional GC Content Calculation ---")
    regional_gc_window = params.get("regional_gc_window", 5)
    regional_gc_overlap = params.get("regional_gc_overlap", 0)
    sirna_regional_GC_raw = [utils.get_regional_gc_content(seq, window_size=regional_gc_window, overlap=regional_gc_overlap) for seq in data['siRNA_seq']]
    num_regional_gc_features = max((len(features) for features in sirna_regional_GC_raw), default=0)
    sirna_regional_GC_df = pd.DataFrame(sirna_regional_GC_raw, index=list(data['siRNA']),
                                        columns=[f'regional_GC_{i}' for i in range(num_regional_gc_features)])
    sirna_regional_GC_df = sirna_regional_GC_df.loc[~sirna_regional_GC_df.index.duplicated(keep='first')]
    sirna_regional_GC_df = sirna_regional_GC_df.fillna(0)
    print("--- Finished Regional GC Content Calculation ---")

    print("--- Starting Terminal Base One-Hot Encoding Calculation ---")
    sirna_terminal_onehot_raw = [utils.get_terminal_base_one_hot(seq, length=sirna_length) for seq in data['siRNA_seq']]
    sirna_terminal_onehot = pd.DataFrame(sirna_terminal_onehot_raw, index=list(data['siRNA']))
    sirna_terminal_onehot.columns = ['5_prime_A', '5_prime_C', '5_prime_G', '5_prime_U',
                                     '3_prime_A', '3_prime_C', '3_prime_G', '3_prime_U']
    sirna_terminal_onehot = sirna_terminal_onehot.loc[~sirna_terminal_onehot.index.duplicated(keep='first')]
    sirna_terminal_onehot = sirna_terminal_onehot.fillna(0)
    print("--- Finished Terminal Base One-Hot Encoding Calculation ---")

    print("--- Starting Thermodynamic Asymmetry Calculation ---")
    asym_5_prime_len = params.get("asym_5_prime_len", 5)
    asym_3_prime_len = params.get("asym_3_prime_len", 5)
    sirna_thermo_asymmetry_raw = [utils.get_thermo_asymmetry(seq, region1_len=asym_5_prime_len, region2_len=asym_3_prime_len) for seq in data['siRNA_seq']]
    sirna_thermo_asymmetry = pd.DataFrame(sirna_thermo_asymmetry_raw, columns=['thermo_asymmetry'], index=data['siRNA'] + '_' + data['mRNA'])
    sirna_thermo_asymmetry = sirna_thermo_asymmetry.loc[~sirna_thermo_asymmetry.index.duplicated(keep='first')]
    sirna_thermo_asymmetry = sirna_thermo_asymmetry.fillna(0)
    print("--- Finished Thermodynamic Asymmetry Calculation ---")

    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data['siRNA']
    sirna_k_mers = sirna_k_mers.loc[~sirna_k_mers.index.duplicated(keep='first')]
    sirna_k_mers = sirna_k_mers.fillna(0)

    sirna_pos_scores = [utils.rules_scores(seq) for seq in data['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    sirna_pos_scores = sirna_pos_scores.loc[~sirna_pos_scores.index.duplicated(keep='first')]
    sirna_pos_scores = sirna_pos_scores.fillna(0)

    print("--- Consolidating siRNA Node Features ---")
    sirna_pd = pd.concat([
        sirna_embedding_df,
        sirna_sfold_feat,
        sirna_GC.reindex(sirna_embedding_df.index).fillna(0),
        sirna_AU.reindex(sirna_embedding_df.index).fillna(0),
        sirna_regional_GC_df.reindex(sirna_embedding_df.index).fillna(0),
        sirna_terminal_onehot.reindex(sirna_embedding_df.index).fillna(0),
        sirna_k_mers.reindex(sirna_embedding_df.index).fillna(0),
        sirna_pos_scores.reindex(sirna_embedding_df.index).fillna(0)
    ], axis=1)
    sirna_pd = sirna_pd.fillna(0)
    print(f"siRNA_pd shape: {sirna_pd.shape}")

    print("--- Consolidating mRNA Node Features ---")
    mrna_pd = pd.concat([
        mrna_embedding_df,
        mrna_sfold_feat,
        mrna_GC.reindex(mrna_embedding_df.index).fillna(0),
        mrna_AU.reindex(mrna_embedding_df.index).fillna(0)
    ], axis=1)
    mrna_pd = mrna_pd.fillna(0)
    print(f"mRNA_pd shape: {mrna_pd.shape}")

    print("--- Consolidating Interaction Node Features ---")
    interaction_pd = pd.concat([
        sirna_thermo_feat,
        con_feat,
        sirna_pos_encoding.reindex(sirna_thermo_feat.index).fillna(0),
        sirna_thermo_asymmetry.reindex(sirna_thermo_feat.index).fillna(0)
    ], axis=1)
    interaction_pd = interaction_pd.fillna(0)
    print(f"interaction_pd shape: {interaction_pd.shape}")

    source = data['siRNA'] + "_" + data['mRNA']
    target_siRNA = data['siRNA']
    target_mRNA = data['mRNA']

    all_my_edges1 = pd.DataFrame({'source': source, 'target': target_siRNA})
    all_my_edges2 = pd.DataFrame({'source': source, 'target': target_mRNA})

    all_my_edges = pd.concat([all_my_edges1, all_my_edges2], ignore_index=True, axis=0)

    print("--- Creating HeteroData object ---")
    data_hetero = HeteroData()

    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)
    print("--- Node features added ---")

    sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}
    print("--- Node index mappings created ---")

    print("--- Creating Edge Indices ---")
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []

    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_idx and \
           row['siRNA'] in sirna_idx and \
           row['mRNA'] in mrna_idx:
            edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])
    print("--- Raw edge lists populated ---")

    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    print("--- Edge indices tensors created ---")

    data_hetero = ToUndirected()(data_hetero)
    print("--- Graph converted to undirected ---")

    train_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)
    dev_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)
    test_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_test.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)

    print("--- Initializing NeighborLoaders ---")
    train_loader = NeighborLoader(data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"], input_nodes=('interaction', train_idx), shuffle=True, subgraph_type='induced', filter_per_worker=False)
    val_loader = NeighborLoader(data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"], input_nodes=('interaction', dev_idx), shuffle=False, subgraph_type='induced', filter_per_worker=False)
    test_loader = NeighborLoader(data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"], input_nodes=('interaction', test_idx), shuffle=False, subgraph_type='induced', filter_per_worker=False)
    print("--- NeighborLoaders created ---")

    train_interactions = set(data_train['siRNA'] + "_" + data_train['mRNA'])
    dev_interactions = set(data_dev['siRNA'] + "_" + data_dev['mRNA'])
    test_interactions = set(data_test['siRNA'] + "_" + data_test['mRNA'])

    train_mask = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.bool)
    dev_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    for i, name in enumerate(interaction_pd.index):
        if name in train_interactions:
            train_mask[i] = True
        elif name in dev_interactions:
            dev_mask[i] = True
        elif name in test_interactions:
            test_mask[i] = True

    data_hetero['interaction'].train_mask = train_mask
    data_hetero['interaction'].dev_mask = dev_mask
    data_hetero['interaction'].test_mask = test_mask

    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_idx:
            labels[interaction_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels
    print("--- Labels and Masks assigned ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=params["dropout"]
    ).to(device)

    data_hetero = data_hetero.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=5e-4)  # Increased weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss() if params["loss"] == "mse" else nn.L1Loss()
    print("--- Model and optimizer initialized ---")

    def train():
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size, device=device)

            out = model(batch.x_dict, batch.edge_index_dict)

            loss = criterion(out[batch_indices].squeeze(-1), batch['interaction'].y[batch_indices])

            l1_lambda = params.get("l1_lambda", 1e-5)
            loss += l1_lambda * model.l1_loss()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def test(loader):
        model.eval()
        preds, truths = [], []
        for i, batch in enumerate(loader):
            batch_size = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size)

            out = model(batch.x_dict, batch.edge_index_dict)
            preds.append(out[batch_indices].cpu())
            truths.append(batch['interaction'].y[batch_indices].cpu())
        return torch.cat(preds), torch.cat(truths)

    best_val_loss = float('inf')
    print("--- Starting training loop ---")
    for epoch in range(params["epochs"]):
        loss = train()

        val_pred, val_true = test(val_loader)
        val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'model_fold{n}.pt')
            print(f"  --> New best model saved for Fold {n} at Epoch {epoch} with Val Loss: {val_loss:.4f}")

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
    print("--- Training loop finished ---")

    model_path = f'model_fold{n}.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"--- Loaded best model for Fold {n} from {model_path} for final evaluation ---")
    else:
        print(f"Error: Best model checkpoint not found for fold {n} at {model_path}. Skipping test for this fold.")
        continue

    with torch.no_grad():
        test_pred, test_true = test(test_loader)
        test_pred = test_pred.cpu().numpy().flatten()
        test_true = test_true.cpu().numpy().flatten()

        valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
        if not np.any(valid_mask):
            print("Warning: All predictions are NaN! Skipping metrics for this fold.")
            continue

        test_pred = test_pred[valid_mask]
        test_true = test_true[valid_mask]

        r_value, _ = scipy.stats.pearsonr(test_true, test_pred)
        score_PCC.append(r_value)
        print("PCC:", r_value)

        spearman = scipy.stats.spearmanr(test_true, test_pred)
        score_SPCC.append(spearman[0])
        print("SPCC:", spearman[0])

        score_mse.append(mean_squared_error(test_true, test_pred))

        if len(np.unique(test_true > 0.7)) > 1:
            score_auc.append(roc_auc_score((test_true > 0.7).astype(int), test_pred))

    if score_PCC:
        print(f"\nFold Metrics for Fold {n}:")
        print(f"PCC: {score_PCC[-1]:.4f}")
        print(f"SPCC: {score_SPCC[-1]:.4f}")
        print(f"MSE: {score_mse[-1]:.4f}")
        if score_auc:
            print(f"AUC: {score_auc[-1]:.4f}")
    print(f"Fold {n} finished!\n{'-'*30}")

print(f"\n{'='*40}\nOverall Cross-Validation Metrics:")
print(f"Overall PCC score = {np.mean(score_PCC):.4f}")
print(f"Overall SPCC score = {np.mean(score_SPCC):.4f}")
print(f"Overall MSE score = {np.mean(score_mse):.4f}")
if score_auc:
    print(f"Overall AUC score = {np.mean(score_auc):.4f}")
else:
    print("AUC score not calculated (not enough true classes for AUC).")