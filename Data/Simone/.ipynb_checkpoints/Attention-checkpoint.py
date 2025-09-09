import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, LayerNorm, GATConv
from torch.nn import BatchNorm1d, Dropout
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
import largeDataUtils as utils1

# Load parameters
params = json.load(open("param_att.json", 'r'))

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]

# Initialize metric lists
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

class ImprovedHeteroGNN(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, node_feature_dims, dropout_rate=0.3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0]
        
        # Input projection layers for each node type to normalize dimensions
        self.input_projections = nn.ModuleDict()
        for node_type in self.node_types:
            if node_type in node_feature_dims:
                self.input_projections[node_type] = Linear(node_feature_dims[node_type], layer_sizes[0])
        
        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else layer_sizes[0]  # Use first layer size instead of -1
            out_channels_i = layer_sizes[i]
            
            # Use attention-based convolution for better representation learning
            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): GATConv(
                    in_channels, out_channels_i, heads=4, concat=False, 
                    dropout=dropout_rate, add_self_loops=False
                ),
                ('mRNA', 'interacts_with', 'interaction'): GATConv(
                    in_channels, out_channels_i, heads=4, concat=False, 
                    dropout=dropout_rate, add_self_loops=False
                ),
                ('interaction', 'rev_interacts_with', 'siRNA'): GATConv(
                    in_channels, out_channels_i, heads=4, concat=False, 
                    dropout=dropout_rate, add_self_loops=False
                ),
                ('interaction', 'rev_interacts_with', 'mRNA'): GATConv(
                    in_channels, out_channels_i, heads=4, concat=False, 
                    dropout=dropout_rate, add_self_loops=False
                ),
            }, aggr='mean')
            
            self.convs.append(conv)
            
            # Use LayerNorm instead of BatchNorm for better stability
            self.norms.append(nn.ModuleDict({
                node_type: LayerNorm(out_channels_i) for node_type in self.node_types
            }))
            
            self.dropouts.append(Dropout(dropout_rate))
        
        # Multi-layer prediction head with residual connections
        self.prediction_head = nn.Sequential(
            Linear(layer_sizes[-1], layer_sizes[-1] // 2),
            nn.LeakyReLU(negative_slope=0.2),
            Dropout(dropout_rate),
            Linear(layer_sizes[-1] // 2, layer_sizes[-1] // 4),
            nn.LeakyReLU(negative_slope=0.2),
            Dropout(dropout_rate),
            Linear(layer_sizes[-1] // 4, out_channels)
        )
        
        # Skip connection projection
        self.skip_projection = Linear(layer_sizes[0], layer_sizes[-1]) if len(layer_sizes) > 1 else None
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.414)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x_dict, edge_index_dict):
        # Apply input projections to standardize dimensions
        for node_type in x_dict:
            if node_type in self.input_projections:
                x_dict[node_type] = self.input_projections[node_type](x_dict[node_type])
        
        # Store initial representation for skip connection
        initial_x = x_dict['interaction'].clone() if self.skip_projection else None
        
        for i, (conv, norm_dict, dropout) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            for node_type in x_dict_new:
                if node_type in norm_dict:
                    x_dict_new[node_type] = norm_dict[node_type](x_dict_new[node_type])
                
                # Use ELU activation for better gradient flow
                x_dict_new[node_type] = F.elu(x_dict_new[node_type])
                x_dict_new[node_type] = dropout(x_dict_new[node_type])
            
            x_dict = x_dict_new
        
        interaction_emb = x_dict['interaction']
        
        # Add skip connection if applicable
        if self.skip_projection is not None and initial_x is not None:
            skip_connection = self.skip_projection(initial_x)
            interaction_emb = interaction_emb + skip_connection
        
        return self.prediction_head(interaction_emb)
    
    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# Efficient training function with gradient clipping
def train_epoch(model, train_loader, optimizer, criterion, device, l1_lambda=1e-5, clip_grad=1.0):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        batch_size = batch['interaction'].num_nodes
        batch_indices = torch.arange(batch_size, device=device)
        
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = criterion(out[batch_indices].squeeze(-1), batch['interaction'].y[batch_indices])
        
        # Add L1 regularization
        if l1_lambda > 0:
            loss += l1_lambda * model.l1_loss()
        
        loss.backward()
        
        # Gradient clipping for stability
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    preds, truths = [], []
    
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch['interaction'].num_nodes
        batch_indices = torch.arange(batch_size, device=device)
        
        out = model(batch.x_dict, batch.edge_index_dict)
        preds.append(out[batch_indices].cpu())
        truths.append(batch['interaction'].y[batch_indices].cpu())
    
    return torch.cat(preds), torch.cat(truths)

print(f"Dropout: {params['dropout']}")
print(f"Learning Rate: {params['lr']}")
print(f"Epochs: {params['epochs']}")

for n in range(10):
    print(f"\n--- Testing Fold {n} ---")

    # Load the single CSV file for testing
    test_file = "siRNA_mRNA_all_data.csv" # <--- IMPORTANT: Adjust this path to your actual test data file
    try:
        data_test = pd.read_csv(test_file)
        print(f"Successfully loaded test data from: {test_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {test_file}. Skipping fold {n}.")
        continue
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}. Skipping fold {n}.")
        continue

    # Prepare sequences for feature calculation
    # MP-RNA Transformer expects 'U', other features may work with 'T' or 'U'
    # Original script converted U to T, which is counterproductive for MP-RNA.
    # We will ensure sequence versions appropriate for each feature are used.
    data_test['siRNA_seq_for_mp_rna'] = data_test['siRNA_seq'].str.replace('T', 'U', regex=False)
    data_test['mRNA_seq_for_mp_rna'] = data_test['mRNA_seq'].str.replace('T', 'U', regex=False)
    data_test['siRNA_seq_for_other_features'] = data_test['siRNA_seq'].str.replace('U', 'T', regex=False) # For compatibility with old utils features
    data_test['mRNA_seq_for_other_features'] = data_test['mRNA_seq'].str.replace('U', 'T', regex=False) # For compatibility with old utils features
    
    print("\n--- Feature Processing ---")


    # print("--- Starting Sequence Embedding (siRNA) ---")
    # sirna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in data_test['siRNA_seq']]
    # sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(data_test['siRNA']))
    # # print(sirna_embedding_df.head(20))
    # sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')]
    # print("--- Finished Sequence Embedding (siRNA) ---")

    # print("--- Starting Sequence Embedding (mRNA) ---")
    # mrna_unique_seq_df = data_test.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
    # mrna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    # mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
    # mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')]
    # print("--- Finished Sequence Embedding (mRNA) ---")
    # # --- End of Sequence Embedding Update ---
    # print(f"\n--- mRNA One-Hot Encoding ---")
    # print(f"Sequence: {first_mRNA_seq_RNA_FM}")
    # print(f"Full DataFrame Shape: {mrna_onehot.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {mrna_onehot.iloc[0].values[:20]}...")
    # --- Feature processing with variable length handling and padding ---

    # 1. One-hot encoding for siRNA (PADDED to MAX_SIRNA_LENGTH)
    sirna_onehot = []
    for seq in data_test['siRNA_seq']:
        sirna_onehot.append(utils1.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data_test['siRNA']))
    # print(f"\n--- siRNA One-Hot Encoding ---")
    # print(f"Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_onehot.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {sirna_onehot.iloc[0].values[:20]}...")

    # 2. mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data_test.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils1.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))

    # 3. Positional encoding (PADDED to MAX_SIRNA_LENGTH * dmodel)
    sirna_pos_encoding = []
    for idx, row in data_test.iterrows():
        mrna_start_pos = max(0, int(row['pos']))
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos,
            len(row['siRNA_seq']),
            MAX_SIRNA_LENGTH,
            params["dmodel"]
        ))
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data_test['siRNA'] + '_' + data_test['mRNA'])
    # print(f"\n--- Positional Encoding ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"mRNA Match Position (from 'pos' column): {first_pos}")
    # print(f"Full DataFrame Shape: {sirna_pos_encoding.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {sirna_pos_encoding.iloc[0].values[:20]}...")

    # 4. Thermodynamics (PADDED to MAX_SIRNA_LENGTH)
    sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data_test['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    temp_interaction_index = data_test['siRNA'] + '_' + data_test['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')
    # print(f"\n--- Thermodynamics Features ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_thermo_feat.shape}")
    # print(f"First Sample's Feature Vector (all values): {sirna_thermo_feat.iloc[0].values}")

    # # Co-fold features
    # try:
    #     con_feat = pd.read_csv("Simone_split_preprocess/con_matrix_Simone_meanSum50.txt", header=None, index_col=0)
    #     con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    # except FileNotFoundError:
    #     print("Warning: co-fold feature file 'Simone_split_preprocess/con_matrix_Simone_meanSum50.txt' not found. Filling with zeros.")
    #     con_feat = pd.DataFrame(0.0, index=sirna_thermo_feat.index, columns=[f'cofold_dim_{i}' for i in range(50)])
    
    # # self-fold features
    # ## siRNA
    # try:
    #     sirna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt", header=None, index_col=0)
    #     sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0)
    # except FileNotFoundError:
    #     print("Warning: siRNA self-fold feature file 'Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt' not found. Filling with zeros.")
    #     sirna_sfold_feat = pd.DataFrame(0.0, index=sirna_embedding_df.index, columns=[f'sfold_siRNA_dim_{i}' for i in range(6)])

    # ## mRNA
    # try:
    #     mrna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt", header=None, index_col=0)
    #     mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0)
    # except FileNotFoundError:
    #     print("Warning: mRNA self-fold feature file 'Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt' not found. Filling with zeros.")
    #     mrna_sfold_feat = pd.DataFrame(0.0, index=mrna_embedding_df.index, columns=[f'sfold_mRNA_dim_{i}' for i in range(100)])

    # Co-fold features
    try:
        con_feat = pd.read_csv("Simone_split_preprocess/con_matrix_Simone_meanSum50.txt", header=None, index_col=0)
        con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    except FileNotFoundError:
        print("Warning: co-fold feature file 'Simone_split_preprocess/con_matrix_Simone_meanSum50.txt' not found. Filling with zeros.")
        con_feat = pd.DataFrame(0.0, index=sirna_thermo_feat.index, columns=[f'cofold_dim_{i}' for i in range(50)])
    
    # self-fold features
    ## siRNA
    try:
        sirna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt", header=None, index_col=0)
        sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index).fillna(0)
    except FileNotFoundError:
        print("Warning: siRNA self-fold feature file 'Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt' not found. Filling with zeros.")
        sirna_sfold_feat = pd.DataFrame(0.0, index=sirna_embedding_df.index, columns=[f'sfold_siRNA_dim_{i}' for i in range(6)])

    ## mRNA
    try:
        mrna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt", header=None, index_col=0)
        mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    except FileNotFoundError:
        print("Warning: mRNA self-fold feature file 'Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt' not found. Filling with zeros.")
        mrna_sfold_feat = pd.DataFrame(0.0, index=mrna_embedding_df.index, columns=[f'sfold_mRNA_dim_{i}' for i in range(100)])

    # # GC percentage
    # sirna_GC = [utils1.countGC(seq) for seq in data_test['siRNA_seq']]
    # sirna_GC = pd.DataFrame(sirna_GC, columns=['GC_content'], index=list(data_test['siRNA']))
    # sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')]

    # mrna_GC = [utils1.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    # mrna_GC = pd.DataFrame(mrna_GC, columns=['GC_content'], index=list(mrna_unique_seq_df['mRNA']))
    # mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')]
    # 8. GC percentage (variable length robust)
    sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data_test['siRNA_seq']], index=list(data_test['siRNA']))
    mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))

    # 9. K-mers (All now return fixed-size lists)
    sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data_test['siRNA']
    # print(f"\n--- siRNA K-mer Frequencies ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_k_mers.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {sirna_k_mers.iloc[0].values[:20]}...")

    # 10. siRNA rules codes (PADDED to 19*3)
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data_test['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data_test['siRNA']))
    # print(f"\n--- siRNA Rules Scores ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_pos_scores.shape}")
    # print(f"First Sample's Feature Vector (all values): {sirna_pos_scores.iloc[0].values}")

    # Node features
    sirna_pd = pd.concat([sirna_onehot,sirna_sfold_feat,sirna_GC,sirna_k_mers,sirna_pos_scores],axis = 1)
    mrna_pd = pd.concat([mrna_onehot,mrna_sfold_feat,mrna_GC],axis = 1)

    # Interactive nodes
    sirna_pos_encoding.index = sirna_thermo_feat.index
    interaction_pd = pd.concat([sirna_thermo_feat,con_feat,sirna_pos_encoding],axis=1)
    
    # Create heterogeneous graph data
    data_hetero = HeteroData()
    
    # Add node features with proper normalization
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)  
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)
    
    # Normalize features
    for node_type in ['siRNA', 'mRNA', 'interaction']:
        x = data_hetero[node_type].x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-8
        data_hetero[node_type].x = (x - mean) / std
    
    # Create mappings and edges (keeping your logic)
    sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}
    
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []
    
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
        edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])
    
    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
    
    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    
    # Create split indices
    interaction_idx = {f"{row.siRNA}_{row.mRNA}": i for i, row in enumerate(data_test.itertuples())}
    
    # train_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train.iterrows()], dtype=torch.long)
    # dev_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev.iterrows()], dtype=torch.long)
    test_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_test.iterrows()], dtype=torch.long)
    
    data_hetero = ToUndirected()(data_hetero)
    
    # Create labels
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        labels[interaction_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels
    
    # # Create data loaders with better configuration
    # train_loader = NeighborLoader(
    #     data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"],
    #     input_nodes=('interaction', train_idx), shuffle=True, subgraph_type='induced',
    #     filter_per_worker=False, num_workers=0  # Avoid multiprocessing issues
    # )
    
    # val_loader = NeighborLoader(
    #     data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"],
    #     input_nodes=('interaction', dev_idx), shuffle=False, subgraph_type='induced',
    #     filter_per_worker=False, num_workers=0
    # )
    
    test_loader = NeighborLoader(
        data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"],
        input_nodes=('interaction', test_idx), shuffle=False, subgraph_type='induced',
        filter_per_worker=False, num_workers=0
    )
    
    # Initialize improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get feature dimensions for proper initialization
    node_feature_dims = {
        'siRNA': data_hetero['siRNA'].x.shape[1],
        'mRNA': data_hetero['mRNA'].x.shape[1], 
        'interaction': data_hetero['interaction'].x.shape[1]
    }
    
    model = ImprovedHeteroGNN(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        node_feature_dims=node_feature_dims,
        dropout_rate=params["dropout"]
    ).to(device)
    
    data_hetero = data_hetero.to(device)
    
    # Improved optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.MSELoss() if params["loss"] == "mse" else nn.SmoothL1Loss()
    
    early_stopping = EarlyStopping(patience=20)
    best_val_loss = float('inf')
    
    # Load the saved weights
    model_path = f"best_model_fold{n}.pt" # <--- IMPORTANT: Ensure this path is correct for your saved models
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot test this fold.")
        continue # Skip to next fold
    except Exception as e:
        print(f"An error occurred while loading model for fold {n}: {e}. Skipping this fold.")
        continue
    
    test_pred, test_true = test(model,test_loader,device)
    test_pred = test_pred.numpy().flatten()
    test_true = test_true.numpy().flatten()

    # Filter out NaNs if any (though features are filled with 0, so should be less likely now)
    valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
    if not np.any(valid_mask):
        print("Warning: All predictions or true values are NaN for this fold. Skipping metrics.")
        continue

    test_pred = test_pred[valid_mask]
    test_true = test_true[valid_mask]

    r_value, _ = scipy.stats.pearsonr(test_true, test_pred)
    score_PCC.append(r_value)
    print(f"PCC: {r_value:.4f}")

    spearman, _ = scipy.stats.spearmanr(test_true, test_pred)
    score_SPCC.append(spearman)
    print(f"SPCC: {spearman:.4f}")

    mse = mean_squared_error(test_true, test_pred)
    score_mse.append(mse)
    print(f"MSE: {mse:.4f}")

    # AUC (Area Under the Receiver Operating Characteristic Curve)
    # This requires binary classification. We'll use a threshold (e.g., 0.7 efficacy) to binarize true labels.
    if len(np.unique(test_true > 0.7)) > 1: # Check if there are at least two unique binary classes
        auc = roc_auc_score((test_true > 0.7).astype(int), test_pred)
        print(f"AUC: {auc:.4f}")
        score_auc.append(auc)
    else:
        print(f"AUC: Not enough unique classes for AUC calculation (all 'efficacy' values are on one side of 0.7 threshold).")

    print(f"Fold {n} finished!")
    
#     # Calculate metrics
#     try:
#         r_value, _ = scipy.stats.pearsonr(test_true, test_pred)
#         score_PCC.append(r_value)
#         print("PCC:", r_value)
        
#         spearman = scipy.stats.spearmanr(test_true, test_pred)
#         score_SPCC.append(spearman[0])
#         print("SPCC:", spearman[0])
        
#         score_mse.append(mean_squared_error(test_true, test_pred))
        
#         if len(np.unique(test_true > 0.7)) > 1:
#             score_auc.append(roc_auc_score((test_true > 0.7).astype(int), test_pred))
#     except Exception as e:
#         print(f"Error calculating metrics: {e}")
#         continue
    
#     print(f"Fold {n} finished!")

# # Print final results
# if score_PCC:
#     print(f"\nFinal Results:")
#     print(f"Overall PCC score = {np.mean(score_PCC):.4f} ± {np.std(score_PCC):.4f}")
#     print(f"Overall SPCC score = {np.mean(score_SPCC):.4f} ± {np.std(score_SPCC):.4f}")
#     print(f"Overall MSE score = {np.mean(score_mse):.4f} ± {np.std(score_mse):.4f}")
#     if score_auc:
#         print(f"Overall AUC score = {np.mean(score_auc):.4f} ± {np.std(score_auc):.4f}")