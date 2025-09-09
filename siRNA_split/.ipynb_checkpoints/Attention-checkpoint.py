import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear
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
import utils  # Features calculation

# Load parameters
params = json.load(open("siRNA_param_pytorch.json", 'r'))

# Initialize metric lists
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

class HeteroGAT(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, heads=4, dropout=0.2):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.heads = heads
        
        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else -1  # -1 lets GATConv infer
            out_channels_i = layer_sizes[i]
            
            # Last layer typically uses 1 head
            heads_i = 1 if i == len(layer_sizes) - 1 else heads
            per_head_out = out_channels_i if heads_i == 1 else out_channels_i // heads_i

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): GATConv(
                    in_channels, per_head_out, heads=heads_i,
                    dropout=dropout, add_self_loops=False),
                ('mRNA', 'interacts_with', 'interaction'): GATConv(
                    in_channels, per_head_out, heads=heads_i,
                    dropout=dropout, add_self_loops=False),
                ('interaction', 'rev_interacts_with', 'siRNA'): GATConv(
                    in_channels, per_head_out, heads=heads_i,
                    dropout=dropout, add_self_loops=False),
                ('interaction', 'rev_interacts_with', 'mRNA'): GATConv(
                    in_channels, per_head_out, heads=heads_i,
                    dropout=dropout, add_self_loops=False),
            }, aggr='mean')
            self.convs.append(conv)
        
        self.lin = Linear(layer_sizes[-1], out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:  # Apply activation and dropout to all but last layer
                x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                         for key, x in x_dict.items()}
        return self.lin(x_dict['interaction'])

print(params["dropout"])
print(params["lr"])
print(params["epochs"])

for n in range(10):
    print(f"Processing fold {n}")
    print("10")
    # Read and preprocess data
    data_train = pd.read_csv(f"siRNA_split_datasets/split{n}/train.csv")
    data_dev = pd.read_csv(f"siRNA_split_datasets/split{n}/dev.csv")
    data_test = pd.read_csv(f"siRNA_split_datasets/split{n}/test.csv")
    
    data_train['split'] = 'train'
    data_dev['split'] = 'dev' 
    data_test['split'] = 'test'
    
    # Substitute U to T
    for df in [data_train, data_dev, data_test]:
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')
    
    data = pd.concat([data_train, data_dev, data_test], axis=0)
    
    # Feature processing (same as original)
    # one-hot
    ## siRNA
    sirna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["sirna_length"]) for seq in data['siRNA_seq']]
    sirna_onehot = pd.DataFrame(sirna_onehot,index=list(data['siRNA']))

    ## mRNA
    mrna_onehot_temp = data.loc[:,['mRNA','mRNA_seq']]
    mrna_onehot_temp = mrna_onehot_temp.drop_duplicates(subset="mRNA")

    mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["max_mrna_len"]) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_onehot = pd.DataFrame(mrna_onehot,index = list(mrna_onehot_temp['mRNA']))

    # Positional encoding
    trans_table = str.maketrans('ATCG', 'TAGC')
    data['match_pos'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
    data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].index(row['match_pos']),axis = 1)

    sirna_pos_encoding = [utils.get_pos_embedding_sequence(num,params["sirna_length"],params["dmodel"]) for num in data['match_pos']]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding,index = list(data['siRNA']))

    # Thermodynamics
    sirna_thermo_feat = [utils.cal_thermo_feature(seq.replace("T","U")) for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    sirna_thermo_feat = pd.concat([data['siRNA'].reset_index(drop=True),
                                 data['mRNA'].reset_index(drop=True),
                                 sirna_thermo_feat],
                                 axis = 1)
    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])

    # Co-fold features
    con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt",header=None,index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index)

    # sel-fold features
    ## siRNA
    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt",header=None,index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)

    ## mRNA
    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt",header=None,index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index)

    # AGO2
    ## siRNA-AGO2
    sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2.csv",index_col = 0)
    sirna_ago = sirna_ago.reindex(sirna_onehot.index)

    ## mRNA-AGO2
    mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2.csv",index_col=0)
    mrna_ago = mrna_ago.reindex(mrna_onehot.index)

    # GC percentage
    ## siRNA
    sirna_GC = [utils.countGC(seq) for seq in data['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC,index=list(data['siRNA']))

    ## mRNA
    mrna_GC = [utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_onehot_temp['mRNA']))

    # k-mers
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

    sirna_k_mers = pd.concat([sirna_1_mer,sirna_2_mers, sirna_3_mers,sirna_4_mers,sirna_5_mers], axis = 1)
    sirna_k_mers.index = data['siRNA']

    # siRNA rules codes
    sirna_pos_scores = [utils.rules_scores(seq) for seq in data['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index = list(data['siRNA']))
    
    # Create node features
    sirna_pd = pd.concat([sirna_onehot,sirna_sfold_feat,sirna_ago,sirna_GC,sirna_k_mers,sirna_pos_scores],axis = 1)
    mrna_pd = pd.concat([mrna_onehot,mrna_sfold_feat,mrna_ago,mrna_GC],axis = 1)
    sirna_pos_encoding.index = sirna_thermo_feat.index
    interaction_pd = pd.concat([sirna_thermo_feat,con_feat,sirna_pos_encoding],axis=1)
    
    # Create heterogeneous graph data
    data_hetero = HeteroData()
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)
    
    # Add edges
    sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}
    
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []
    
    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
        edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])
    
    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
    
    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    
    # Split data into train/dev/test
    train_mask = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.bool)
    dev_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    
    interaction_idx = {f"{row.siRNA}_{row.mRNA}": i for i, row in enumerate(data.itertuples())}
    
    train_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train.iterrows()], dtype=torch.long)
    dev_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev.iterrows()], dtype=torch.long)
    test_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_test.iterrows() ], dtype=torch.long)
    
    data_hetero = ToUndirected()(data_hetero)
    
    # Create masks
    train_interactions = set(data_train['siRNA'] + "_" + data_train['mRNA'])
    dev_interactions = set(data_dev['siRNA'] + "_" + data_dev['mRNA'])
    test_interactions = set(data_test['siRNA'] + "_" + data_test['mRNA'])
    
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
    
    # Create labels tensor
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        labels[interaction_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels
    
    # Create data loaders
    train_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', train_idx),
        shuffle=True,
        subgraph_type='induced',
        filter_per_worker=False
    )
    
    val_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', dev_idx),
        shuffle=False,
        subgraph_type='induced',
        filter_per_worker=False
    )

    test_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', test_idx),
        shuffle=False,
        subgraph_type='induced',
        filter_per_worker=False
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGAT(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        heads=4,  # Number of attention heads
        dropout=params["dropout"]
    ).to(device)
    
    data_hetero = data_hetero.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss() if params["loss"] == "mse" else nn.L1Loss()
    
    # Training loop
    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_size = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size, device=device)
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = criterion(out[batch_indices].squeeze(-1), 
                          batch['interaction'].y[batch_indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
     
    @torch.no_grad()
    def test(loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            batch_size = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size, device=device)
            out = model(batch.x_dict, batch.edge_index_dict)
            preds.append(out[batch_indices].cpu())
            truths.append(batch['interaction'].y[batch_indices].cpu())
        return torch.cat(preds), torch.cat(truths)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(params["epochs"]):
        loss = train()
        
        # Validation
        val_pred, val_true = test(val_loader)
        val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f'best_model_fold{n}.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'best_model_fold{n}.pt'))
    with torch.no_grad():
        test_pred, test_true = test(test_loader)
        test_pred = test_pred.numpy().flatten()
        test_true = test_true.numpy().flatten()
    
        # Filter invalid values
        valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
        if np.any(valid_mask):
            test_pred = test_pred[valid_mask]
            test_true = test_true[valid_mask]
            
            # Calculate metrics
            r_value,_ = scipy.stats.pearsonr(test_true, test_pred)
            score_PCC.append(r_value)
            print("PCC:", r_value)
            
            spearman = scipy.stats.spearmanr(test_true, test_pred)
            score_SPCC.append(spearman[0])
            print("SPCC:", spearman[0])
        
            score_mse.append(mean_squared_error(test_true, test_pred))
        
            if len(np.unique(test_true > 0.7)) > 1:
                score_auc.append(roc_auc_score((test_true > 0.7).astype(int), test_pred))

    # Final output with validation
    if score_PCC:
        print(f"\nFold Metrics:")
        print(f"PCC: {score_PCC[-1]:.4f}")
        print(f"SPCC: {score_SPCC[-1]:.4f}")
        print(f"MSE: {score_mse[-1]:.4f}")
        if score_auc: 
            print(f"AUC: {score_auc[-1]:.4f}")
    
    print(f"Fold {n} finished!")

# Print final metrics
print("Overall PCC score =", np.mean(score_PCC))
print("Overall SPCC score =", np.mean(score_SPCC))
print("Overall MSE score =", np.mean(score_mse))
print("Overall AUC score =", np.mean(score_auc))