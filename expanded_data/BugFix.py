import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import scipy.stats
import json
import os # For checking file existence

# Import the revised utils1.py
import BugUtils as utils1 

# --- Load parameters ---
try:
    with open("siRNA_param_pytorch.json", 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print("siRNA_param_pytorch.json not found. Creating a default one.")
    params = {
        "dmodel": 6,
        "sirna_length": 21, # This will now act as MAX_SIRNA_LENGTH for padding
        "max_mrna_len": 9756, # Example, adjust based on your actual mRNA max length
        "batch_size": 16,
        "epochs": 26,
        "dropout": 0.6489414323496209,
        "lr": 0.003824895684368415,
        "loss": "mse",
        "heads": 4, 
        "hinsage_layer_sizes": [64, 32], # Default hidden layer sizes for SAGEConv
        "l1_lambda": 1e-5, # Default L1 regularization strength
        "hop_samples": [20, 10] # Example hop samples for NeighborLoader
    }
    with open("siRNA_param_pytorch.json", 'w') as f:
        json.dump(params, f, indent=4)


# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"] 

# --- Initialize metric lists ---
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

# --- Hetero-SAGE Model Definition ---
class HeteroSAGE(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0] # Get node types from metadata

        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else -1 
            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('mRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'siRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'mRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
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
            
            # Apply LeakyReLU and Dropout after BatchNorm for each node type
            for node_type in x_dict:
                if node_type in bn_dict: 
                    # --- FIX FOR ValueError: Expected more than 1 value per channel ---
                    # Only apply BatchNorm if batch size for this node type is > 1 or if model is in eval mode
                    if x_dict[node_type].size(0) > 1 or not self.training:
                        x_dict[node_type] = bn_dict[node_type](x_dict[node_type])
                    # Else (if batch size is 1 AND training), skip BatchNorm. This is a common workaround.
                
                x_dict[node_type] = F.leaky_relu(x_dict[node_type])
                x_dict[node_type] = dropout(x_dict[node_type])
        return self.lin(x_dict['interaction'])

    def l1_loss(self):
        """Calculates L1 regularization loss for all model parameters."""
        return sum(p.abs().sum() for p in self.parameters())

print(f"Loaded Parameters: {params}")
print(f"Max siRNA Length for Padding: {MAX_SIRNA_LENGTH}")

# --- K-Fold Cross Validation Loop ---
# Set to 1 fold for demonstration. Change to K for K-Fold CV.
NUM_FOLDS = 1 
for n in range(NUM_FOLDS): 
    print(f"\nProcessing fold {n}")
    
    # Read and preprocess data
    split_dir = f"siRNA_split_datasets/split{n}/"
    if not os.path.exists(split_dir):
        print(f"Error: Directory '{split_dir}' not found. Please ensure your data splits are correctly organized.")
        print("Skipping fold processing.")
        continue

    try:
        data_train = pd.read_csv(os.path.join(split_dir, "train.csv"))
        data_dev = pd.read_csv(os.path.join(split_dir, "dev.csv"))
        data_test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    except FileNotFoundError as e:
        print(f"Error: Data file not found in '{split_dir}'. {e}")
        print("Skipping fold processing.")
        continue
    
    data_train['split'] = 'train'
    data_dev['split'] = 'dev'
    data_test['split'] = 'test'
    
    # Substitute U to T for siRNA_seq to match mRNA_seq (DNA-like) for positional lookup.
    # Note: K-mer, thermo, and rules_scores functions in utils will convert T to U back internally for RNA calculations.
    for df in [data_train, data_dev, data_test]:
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')
    
    data = pd.concat([data_train, data_dev, data_test], axis=0).reset_index(drop=True)
    
    print("\n--- Feature Processing ---")
    
    # Get the first sample's data for printing
    first_sample = data.iloc[0]
    first_siRNA_seq = first_sample['siRNA_seq']
    first_mRNA_seq_RNA_FM = data.loc[data['mRNA'] == first_sample['mRNA'], 'mRNA_seq_RNA-FM'].iloc[0] # Get mRNA_seq_RNA-FM for the first mRNA
    first_mRNA_seq = first_sample['mRNA_seq'] # Original mRNA DNA sequence
    first_pos = first_sample['pos']


    # --- Feature processing with variable length handling and padding ---
    
    # 1. One-hot encoding for siRNA (PADDED to MAX_SIRNA_LENGTH)
    sirna_onehot = []
    for seq in data['siRNA_seq']:
        sirna_onehot.append(utils1.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data['siRNA']))
    print(f"\n--- siRNA One-Hot Encoding ---")
    print(f"Sequence: {first_siRNA_seq}")
    print(f"Full DataFrame Shape: {sirna_onehot.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {sirna_onehot.iloc[0].values[:20]}...")


    # 2. mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA") # Use RNA-FM version here
    mrna_onehot = [utils1.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"]) 
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    print(f"\n--- mRNA One-Hot Encoding ---")
    print(f"Sequence: {first_mRNA_seq_RNA_FM}")
    print(f"Full DataFrame Shape: {mrna_onehot.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {mrna_onehot.iloc[0].values[:20]}...")


    # 3. Positional encoding (PADDED to MAX_SIRNA_LENGTH * dmodel)
    # DIRECTLY use the 'pos' column for the mRNA start position
    sirna_pos_encoding = []
    for idx, row in data.iterrows():
        # Ensure 'pos' is treated as an integer and is non-negative
        mrna_start_pos = max(0, int(row['pos'])) 
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos, 
            len(row['siRNA_seq']),        # Actual siRNA length for PE calculation
            MAX_SIRNA_LENGTH,             # Max length for PE padding
            params["dmodel"]
        ))
    # Align index of positional encoding with interaction nodes
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data['siRNA'] + '_' + data['mRNA'])
    print(f"\n--- Positional Encoding ---")
    print(f"siRNA Sequence: {first_siRNA_seq}")
    print(f"mRNA Match Position (from 'pos' column): {first_pos}")
    print(f"Full DataFrame Shape: {sirna_pos_encoding.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {sirna_pos_encoding.iloc[0].values[:20]}...")


    # 4. Thermodynamics (PADDED to MAX_SIRNA_LENGTH)
    sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH) 
                         for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    
    # Create the interaction index before setting it
    temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')
    print(f"\n--- Thermodynamics Features ---")
    print(f"siRNA Sequence: {first_siRNA_seq}")
    print(f"Full DataFrame Shape: {sirna_thermo_feat.shape}")
    print(f"First Sample's Feature Vector (all values): {sirna_thermo_feat.iloc[0].values}") # Thermo features are usually shorter


    # 5. Co-fold features (using base_pair_probs as indicated by user)
    # Assuming base_pair_probs contains the features for interaction nodes
    con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0) # Reindex to ensure alignment
    print(f"\n--- Co-fold Features (from base_pair_probs.txt) ---")
    print(f"Interaction: {first_sample['siRNA']}_{first_sample['mRNA']}")
    print(f"Full DataFrame Shape: {con_feat.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {con_feat.iloc[0].values[:20]}...")

        
    # 6. Self-fold features (siRNA)
    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    print(f"\n--- siRNA Self-Fold Features ---")
    print(f"siRNA: {first_siRNA_seq}")
    print(f"Full DataFrame Shape: {sirna_sfold_feat.shape}")
    print(f"First Sample's Feature Vector (all values): {sirna_sfold_feat.iloc[0].values}")


    # 7. Self-fold features (mRNA)
    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    print(f"\n--- mRNA Self-Fold Features ---")
    print(f"mRNA: {first_mRNA_seq_RNA_FM}")
    print(f"Full DataFrame Shape: {mrna_sfold_feat.shape}")
    print(f"First Sample's Feature Vector (all values): {mrna_sfold_feat.iloc[0].values}")


    # 8. GC percentage (variable length robust)
    sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data['siRNA_seq']], index=list(data['siRNA']))
    mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))
    print(f"\n--- siRNA GC Percentage ---")
    print(f"siRNA Sequence: {first_siRNA_seq}")
    print(f"Full DataFrame Shape: {sirna_GC.shape}")
    print(f"First Sample's Feature Value: {sirna_GC.iloc[0].values}")
    print(f"\n--- mRNA GC Percentage ---")
    print(f"mRNA Sequence: {first_mRNA_seq_RNA_FM}")
    print(f"Full DataFrame Shape: {mrna_GC.shape}")
    print(f"First Sample's Feature Value: {mrna_GC.iloc[0].values}")


    # 9. K-mers (All now return fixed-size lists)
    sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']]) # MAX_SIRNA_LENGTH passed but ignored in utils1
    sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data['siRNA']
    print(f"\n--- siRNA K-mer Frequencies ---")
    print(f"siRNA Sequence: {first_siRNA_seq}")
    print(f"Full DataFrame Shape: {sirna_k_mers.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {sirna_k_mers.iloc[0].values[:20]}...")


    # 10. siRNA rules codes (PADDED to 19*3)
    SIRNA_RULES_LENGTH = 19 
    sirna_pos_scores = []
    for seq in data['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    print(f"\n--- siRNA Rules Scores ---")
    print(f"siRNA Sequence: {first_siRNA_seq}")
    print(f"Full DataFrame Shape: {sirna_pos_scores.shape}")
    print(f"First Sample's Feature Vector (all values): {sirna_pos_scores.iloc[0].values}")
    

    print("\n--- Assembling GNN Node Features ---")

    # siRNA nodes features
    sirna_pd = pd.concat([sirna_onehot, sirna_sfold_feat, sirna_GC, sirna_k_mers, sirna_pos_scores], axis=1)
    print(f"\n--- Combined siRNA Node Features ---")
    print(f"Full DataFrame Shape: {sirna_pd.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {sirna_pd.iloc[0].values[:20]}...")

    
    # mRNA nodes features
    mrna_pd = pd.concat([mrna_onehot, mrna_sfold_feat, mrna_GC], axis=1)
    print(f"\n--- Combined mRNA Node Features ---")
    print(f"Full DataFrame Shape: {mrna_pd.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {mrna_pd.iloc[0].values[:20]}...")

    # Interaction nodes features
    interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding], axis=1)
    print(f"\n--- Combined Interaction Node Features ---")
    print(f"Full DataFrame Shape: {interaction_pd.shape}")
    print(f"First Sample's Feature Vector (first 20 values): {interaction_pd.iloc[0].values[:20]}...")


    # --- Create heterogeneous graph data structure ---
    data_hetero = HeteroData()
    
    # Add node features (ensure they are aligned by index)
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)
    
    print(f"\n--- PyG HeteroData Node Feature Shapes (Tensor) ---")
    print(f"data_hetero['siRNA'].x shape: {data_hetero['siRNA'].x.shape}")
    print(f"data_hetero['mRNA'].x shape: {data_hetero['mRNA'].x.shape}")
    print(f"data_hetero['interaction'].x shape: {data_hetero['interaction'].x.shape}")

    # Create mapping from node names (from DataFrame indexes) to numerical indices
    sirna_name_to_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_name_to_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_name_to_idx = {name: i for i, name in enumerate(interaction_pd.index)}
    
    # Create edge indices based on name-to-index mappings
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []
    
    for _, row in data.iterrows():
        siRNA_name = row['siRNA']
        mRNA_name = row['mRNA']
        interaction_name = f"{siRNA_name}_{mRNA_name}"

        # Only add edges if all corresponding nodes actually exist in the feature DataFrames
        if (siRNA_name in sirna_name_to_idx and 
            mRNA_name in mrna_name_to_idx and 
            interaction_name in interaction_name_to_idx):
            
            edge_index_siRNA_to_interaction.append([sirna_name_to_idx[siRNA_name], interaction_name_to_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_name_to_idx[mRNA_name], interaction_name_to_idx[interaction_name]])
        else:
            print(f"Warning: Skipping edge for {interaction_name} due to missing node in feature dataframes. Check data consistency.")
            
    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
        
    # Manually add reverse edges
    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    
    print(f"\n--- PyG HeteroData Edge Index Shapes ---")
    for edge_type in data_hetero.edge_types:
        print(f"data_hetero{edge_type}.edge_index shape: {data_hetero[edge_type].edge_index.shape}")

    # Create train/dev/test masks and labels
    train_idx_list = []
    dev_idx_list = []
    test_idx_list = []

    for _, row in data_train.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            train_idx_list.append(interaction_name_to_idx[interaction_name])
    
    for _, row in data_dev.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            dev_idx_list.append(interaction_name_to_idx[interaction_name])
            
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            test_idx_list.append(interaction_name_to_idx[interaction_name])
            
    train_idx = torch.tensor(train_idx_list, dtype=torch.long)
    dev_idx = torch.tensor(dev_idx_list, dtype=torch.long)
    test_idx = torch.tensor(test_idx_list, dtype=torch.long)

    # Create labels tensor for all 'interaction' nodes
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx: 
            labels[interaction_name_to_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels
    
    print(f"\n--- PyG HeteroData Label Shape ---")
    print(f"data_hetero['interaction'].y shape: {data_hetero['interaction'].y.shape}")

    # --- Initialize NeighborLoaders for sampling ---
    # Ensure params["hop_samples"] is defined in your JSON
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

    # --- Model Initialization and Training Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=params["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4) # L2 via weight_decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss() if params["loss"] == "mse" else nn.L1Loss()

    # --- Training Function ---
    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x_dict, batch.edge_index_dict)
            
            loss = criterion(out.squeeze(-1), batch['interaction'].y)

            l1_lambda = params.get("l1_lambda", 1e-5)
            loss += l1_lambda * model.l1_loss()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    # --- Test/Validation Function ---
    @torch.no_grad()
    def test(loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            preds.append(out.cpu())
            truths.append(batch['interaction'].y.cpu())
        return torch.cat(preds), torch.cat(truths)

    # --- Main Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(params["epochs"]):
        train_loss = train()
        
        val_pred, val_true = test(val_loader)
        val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()

        scheduler.step(val_loss)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_fold{n}.pt')
            
        if epoch % 10 == 0: # Print every 10 epochs
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # --- Evaluation on Test Set ---
    model.load_state_dict(torch.load(f'best_model_fold{n}.pt'))
    with torch.no_grad():
        test_pred_tensor, test_true_tensor = test(test_loader)
        test_pred = test_pred_tensor.cpu().numpy().flatten()
        test_true = test_true_tensor.cpu().numpy().flatten()
        
        valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
        if not np.any(valid_mask):
            print("Warning: No valid predictions for metrics calculation in this fold!")
            continue
            
        test_pred = test_pred[valid_mask]
        test_true = test_true[valid_mask]
        
        r_value, _ = scipy.stats.pearsonr(test_true, test_pred)
        score_PCC.append(r_value)
        print(f"Fold {n} PCC: {r_value:.4f}")
        
        spearman = scipy.stats.spearmanr(test_true, test_pred)
        score_SPCC.append(spearman[0])
        print(f"Fold {n} SPCC: {spearman[0]:.4f}")
        
        mse_value = mean_squared_error(test_true, test_pred)
        score_mse.append(mse_value)
        print(f"Fold {n} MSE: {mse_value:.4f}")
        
        binary_true = (test_true > 0.7).astype(int)
        if len(np.unique(binary_true)) > 1: 
            auc_value = roc_auc_score(binary_true, test_pred)
            score_auc.append(auc_value)
            print(f"Fold {n} AUC: {auc_value:.4f}")
        else:
            print(f"Fold {n} AUC not calculated: Not enough unique binary classes for ROC AUC (all labels {'<=0.7' if np.all(binary_true == 0) else '>0.7'}).")

    print(f"--- Fold {n} finished! ---")

# --- Final Overall Metrics Summary ---
print("\n--- Overall Cross-Validation Metrics ---")
if score_PCC:
    print(f"Average PCC: {np.mean(score_PCC):.4f} +/- {np.std(score_PCC):.4f}")
if score_SPCC:
    print(f"Average SPCC: {np.mean(score_SPCC):.4f} +/- {np.std(score_SPCC):.4f}")
if score_mse:
    print(f"Average MSE: {np.mean(score_mse):.4f} +/- {np.std(score_mse):.4f}")
if score_auc:
    print(f"Average AUC: {np.mean(score_auc):.4f} +/- {np.std(score_auc):.4f}")
else:
    print("AUC not calculated across all folds due to insufficient binary labels in some splits.")
