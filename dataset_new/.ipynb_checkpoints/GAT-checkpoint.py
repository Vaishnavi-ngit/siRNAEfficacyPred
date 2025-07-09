import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" # Adjust as needed for your GPU setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GraphConv, Linear
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected # Added for explicit graph transformation
import pandas as pd
import numpy as np
import json
import time # Added for timing epochs

# --- Set manual seeds for reproducibility ---
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42) # For numpy operations

# Make sure utils1.py is in the same directory or accessible via PYTHONPATH
import utils1

with open("Testing.json", 'r') as f:
    params = json.load(f)

MAX_SIRNA_LENGTH = params["sirna_length"]

# --- HIN-SAGE Model Class (with weight initialization) ---
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
        
        # --- Improvement: Initialize weights ---
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier uniform initialization for linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            # Standard initialization for BatchNorm
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # GraphConv layers' internal linear layers are often handled by their own constructors
        # or will inherit from generic nn.Linear initializers if not explicitly defined.


    def forward(self, x_dict, edge_index_dict):
        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                # Robust BatchNorm application: check if node_type exists and tensor is not empty
                if node_type in bn_dict and x_dict[node_type].size(0) > 0:
                    # Apply BatchNorm only if batch size > 1 (during training) or always in eval mode
                    if x_dict[node_type].size(0) > 1 or not self.training:
                        x_dict[node_type] = bn_dict[node_type](x_dict[node_type])
                x_dict[node_type] = F.leaky_relu(x_dict[node_type])
                x_dict[node_type] = dropout(x_dict[node_type])
        return self.lin(x_dict['interaction'])

    def l1_loss(self):
        # Calculates L1 regularization over all model parameters
        return sum(p.abs().sum() for p in self.parameters())

print(f"Loaded Parameters: {params}")
print(f"Max siRNA Length for Padding: {MAX_SIRNA_LENGTH}")

NUM_FOLDS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for GNN operations: {device}")

# --- Pre-loading global feature matrices (Optimization: done once outside loop) ---
# Assuming these files contain features for ALL possible nodes and are static.
# Your provided code loads these *inside* the loop for each fold but from a global path.
# Moving this outside avoids redundant I/O.
print("\n--- Pre-loading global feature matrices ---")
try:
    # IMPORTANT: Ensure these paths are correct relative to where you run the script.
    # Based on your provided code, they are in 'siRNA_split_preprocess/'.
    global_con_feat = pd.read_csv("siRNA_split_preprocess/full_con_matrix.txt", header=None, index_col=0)
    global_sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
    global_mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
    print("Pre-loading successful.")
except FileNotFoundError as e:
    print(f"Error loading global feature matrix: {e}. Please ensure files are in 'siRNA_split_preprocess/'.")
    exit() # Exit if critical files are missing


for n in range(NUM_FOLDS):
    print(f"\nProcessing fold {n}")

    split_dir = f"siRNA_split_datasets/split{n}/"
    
    data_train = pd.read_csv(os.path.join(split_dir, "train.csv"))
    data_dev = pd.read_csv(os.path.join(split_dir, "dev.csv"))

    # Replace 'U' with 'T' (using regex=False for literal replacement)
    for df in [data_train, data_dev]:
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T', regex=False)
        df['mRNA_seq'] = df['mRNA_seq'].str.replace('U', 'T', regex=False) # Ensure mRNA sequences are also processed

    data = pd.concat([data_train, data_dev], axis=0).reset_index(drop=True)

    # --- Feature Processing (Your original logic, untouched) ---
    # Sequences Embedding
    print("--- Starting Sequence Embedding (siRNA) ---")
    siRNA_unique_seq_df = data.loc[:, ['siRNA', 'siRNA_seq']].drop_duplicates(subset="siRNA")
    sirna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in siRNA_unique_seq_df['siRNA_seq']]
    sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(siRNA_unique_seq_df['siRNA']))
    sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')]
    print("--- Finished Sequence Embedding (siRNA) ---")

    print("--- Starting Sequence Embedding (mRNA) ---")
    mrna_unique_seq_df = data.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
    mrna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
    mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')]
    print("--- Finished Sequence Embedding (mRNA) ---")
    
    # Positional Encoding
    sirna_pos_encoding = []
    for _, row in data.iterrows():
        mrna_start_pos = max(0, int(row['pos']))
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos,
            len(row['siRNA_seq']),
            MAX_SIRNA_LENGTH,
            params["dmodel"]
        ))
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data['siRNA'] + '_' + data['mRNA'])

    # Thermodynamics
    sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')

    # Co-fold features (reindexed from pre-loaded global data)
    con_feat = global_con_feat.reindex(sirna_thermo_feat.index).fillna(0) # .fillna(0) for robustness

    # Self-fold features siRNA (reindexed from pre-loaded global data)
    sirna_sfold_feat = global_sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0) # .fillna(0) for robustness

    # Self-fold features mRNA (reindexed from pre-loaded global data)
    mrna_sfold_feat = global_mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0) # .fillna(0) for robustness

    # GC percentage
    sirna_GC = [utils1.countGC(seq) for seq in data['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC, columns=['GC_content'], index=list(data['siRNA']))
    sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')]

    mrna_GC = [utils1.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, columns=['GC_content'], index=list(mrna_unique_seq_df['mRNA']))
    mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')]

    # K-mers
    sirna_1_mer = pd.DataFrame([utils1.single_freq(seq) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils1.double_freq(seq) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq) for seq in data['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data['siRNA']
    sirna_k_mers = sirna_k_mers.loc[~sirna_k_mers.index.duplicated(keep='first')]

    # siRNA rules scores
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    sirna_pos_scores = sirna_pos_scores.loc[~sirna_pos_scores.index.duplicated(keep='first')]

    print("\n--- Assembling GNN Node Features ---")

    # --- Improvement: Use .fillna(0) and .copy() for robustness ---
    sirna_pd = pd.concat([sirna_embedding_df, sirna_sfold_feat, sirna_GC, sirna_k_mers, sirna_pos_scores], axis=1).fillna(0).copy()
    mrna_pd = pd.concat([mrna_embedding_df, mrna_sfold_feat, mrna_GC], axis=1).fillna(0).copy()
    interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding], axis=1).fillna(0).copy()

    # --- Graph Construction (Your original logic, with added checks) ---
    data_hetero = HeteroData()

    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)

    sirna_name_to_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_name_to_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_name_to_idx = {name: i for i, name in enumerate(interaction_pd.index)}

    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []

    for _, row in data.iterrows():
        siRNA_name = row['siRNA']
        mRNA_name = row['mRNA']
        interaction_name = f"{siRNA_name}_{mRNA_name}"

        # --- Improvement: Only add edges if all participating nodes have features ---
        if (siRNA_name in sirna_name_to_idx and 
            mRNA_name in mrna_name_to_idx and 
            interaction_name in interaction_name_to_idx):
            edge_index_siRNA_to_interaction.append([sirna_name_to_idx[siRNA_name], interaction_name_to_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_name_to_idx[mRNA_name], interaction_name_to_idx[interaction_name]])
        else:
            print(f"Warning: Skipping edge for {interaction_name} due to missing features for siRNA:{siRNA_name} or mRNA:{mRNA_name} or interaction itself.")


    if not edge_index_siRNA_to_interaction or not edge_index_mRNA_to_interaction:
        print(f"Error: No valid edges could be created for fold {n}. Skipping this fold.")
        continue # Skip to next fold if no edges

    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

    # --- Improvement: Explicitly make graph undirected ---
    # This also correctly sets the reverse edges if not already done.
    data_hetero = ToUndirected()(data_hetero)
    # The lines below are now redundant if ToUndirected() is used, but keeping them commented if you prefer explicit creation
    # data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    # data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])


    train_idx_list = []
    dev_idx_list = []

    # --- Improvement: Only add indices if the interaction exists in the graph ---
    for _, row in data_train.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            train_idx_list.append(interaction_name_to_idx[interaction_name])
        # else: print(f"Warning: Training interaction {interaction_name} not found in graph.")

    for _, row in data_dev.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            dev_idx_list.append(interaction_name_to_idx[interaction_name])
        # else: print(f"Warning: Dev interaction {interaction_name} not found in graph.")
    
    if not train_idx_list:
        print(f"Warning: No valid training interactions found for fold {n}. Skipping fold.")
        continue
    if not dev_idx_list:
        print(f"Warning: No valid development interactions found for fold {n}. Skipping fold.")
        continue


    train_idx = torch.tensor(train_idx_list, dtype=torch.long)
    dev_idx = torch.tensor(dev_idx_list, dtype=torch.long)

    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    # --- Improvement: Populate labels only for existing interactions ---
    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            labels[interaction_name_to_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels

    train_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', train_idx),
        shuffle=True,
        subgraph_type='induced',
        filter_per_worker=params.get("filter_per_worker", True), # --- Improvement: Can be True for performance ---
        num_workers=params.get("num_workers", 0) # --- Improvement: Set number of workers (0 for debugging) ---
    )

    val_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', dev_idx),
        shuffle=False, # Keep False for validation consistency
        subgraph_type='induced',
        filter_per_worker=params.get("filter_per_worker", True),
        num_workers=params.get("num_workers", 0)
    )

    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=params["dropout"]
    ).to(device)

    # --- Improvement: Use AdamW for better weight decay ---
    # Also, get weight_decay from params with a default
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 1e-4))
    
    # --- Improvement: Enhanced Learning Rate Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=params.get("scheduler_factor", 0.5), # Default 0.5
        patience=params.get("scheduler_patience", 10), # Increased patience for plateaus
        verbose=True,
        min_lr=params.get("min_lr", 1e-6) # Prevents LR from going too low
    )
    
    # --- Improvement: Use Huber Loss as an option for robustness to outliers ---
    if params.get("loss_type", "mse") == "huber":
        criterion = nn.HuberLoss(delta=params.get("huber_delta", 1.0))
        print(f"Using Huber Loss with delta={params.get('huber_delta', 1.0)}")
    elif params["loss"] == "mse":
        criterion = nn.MSELoss()
        print("Using MSE Loss")
    else: # Default to L1Loss if "loss" is not "mse" or "huber"
        criterion = nn.L1Loss()
        print("Using L1 Loss")


    def train():
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x_dict, batch.edge_index_dict)
            
            # --- Improvement: Ensure output matches target shape ---
            loss = criterion(out.squeeze(-1), batch['interaction'].y)

            # L1 regularization (optional, usually weight_decay in AdamW is sufficient)
            l1_lambda = params.get("l1_lambda", 0.0) # Default to 0.0 unless explicitly set
            if l1_lambda > 0:
                loss += l1_lambda * model.l1_loss()

            loss.backward()
            
            # --- Improvement: Gradient Clipping to prevent exploding gradients ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.get("clip_grad_norm", 1.0))
            
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(loader):
        model.eval()
        total_loss = 0
        preds, truths = [], [] # Keep for collecting predictions/truths if needed later (e.g., for R2)
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            
            # --- Improvement: Squeeze output for consistent loss calculation ---
            loss = criterion(out.squeeze(-1), batch['interaction'].y)
            total_loss += loss.item()

            preds.append(out.cpu().squeeze(-1))
            truths.append(batch['interaction'].y.cpu())
        
        # --- Improvement: Return average loss for scheduler.step ---
        avg_val_loss = total_loss / len(loader)
        return avg_val_loss, torch.cat(preds), torch.cat(truths)


    best_val_loss = float('inf')
    epochs_no_improve = 0 # Counter for early stopping
    patience = params.get("early_stopping_patience", 20) # --- Improvement: Early stopping patience ---

    for epoch in range(params["epochs"]):
        start_time = time.time()
        train_loss = train()

        # --- Improvement: Get average validation loss directly ---
        val_loss, val_pred_for_metrics, val_true_for_metrics = validate(val_loader) 
        
        scheduler.step(val_loss) # Step the scheduler with the average validation loss

        # --- Improvement: Early stopping logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Only save if there's an actual improvement
            torch.save(model.state_dict(), f'best_model_fold{n}.pt')
            epochs_no_improve = 0 # Reset counter
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Best), Time: {time.time()-start_time:.2f}s')
        else:
            epochs_no_improve += 1
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, Time: {time.time()-start_time:.2f}s')
            if epochs_no_improve >= patience: # Use >= for clarity
                print(f'Early stopping triggered for fold {n} after {patience} epochs without improvement in validation loss.')
                break # Stop training for this fold

print("\n--- Training and Validation Complete Across All Folds ---")
print("Model checkpoints saved based on best validation loss for each fold.")