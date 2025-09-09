import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
import scipy.stats
import json
import math
import re
import utils1 as utils # Make sure utils.py is in the same directory or accessible via PYTHONPATH

# Load parameters
params = json.load(open("siRNA_param_pytorch.json", 'r'))

# Initialize metric lists
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

# --- HIN-SAGE Model Class (Identical to your training script) ---
class HeteroSAGE(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0] # (node_types, edge_types)

        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else -1 # -1 lets SAGEConv infer
            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('mRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'siRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'mRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
            }, aggr='mean')
            self.convs.append(conv)

            # BatchNorm for each node type
            self.bns.append(nn.ModuleDict({
                node_type: BatchNorm1d(out_channels_i) for node_type in self.node_types
            }))

            self.dropouts.append(Dropout(dropout_rate))

        self.lin = Linear(layer_sizes[-1], out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                # Apply BatchNorm only if the node_type exists in the current bn_dict
                if node_type in bn_dict:
                    x_dict[node_type] = bn_dict[node_type](x_dict[node_type])
                x_dict[node_type] = F.leaky_relu(x_dict[node_type])
                x_dict[node_type] = dropout(x_dict[node_type])
        return self.lin(x_dict['interaction'])

    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())

print(f"Dropout: {params['dropout']}")
print(f"Learning Rate: {params['lr']}")
print(f"Epochs: {params['epochs']}")
print(f"HINSAGE Layer Sizes: {params['hinsage_layer_sizes']}")
print(f"Hop Samples: {params['hop_samples']}")

# Set device for PyTorch operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for GNN operations: {device}")

# --- IMPORTANT: MP-RNA Transformer is loaded globally in utils.py ---
# It will default to CPU unless you modify utils.py to move it to CUDA.
# if utils.GLOBAL_MP_RNA_MODEL is not None:
#     print("MP-RNA Transformer model globally initialized in utils.py (likely on CPU).")
# else:
#     print("WARNING: MP-RNA Transformer model not loaded in utils.py. Its features will be all zeros.")

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


    '''
    Feature processing
    '''

    # --- REPLACED: MP-RNA Transformer Embeddings (instead of one-hot) ---
    print("Generating MP-RNA Transformer embeddings...")
    sirna_transformer_embedding = [
        utils.get_mp_rna_sequence_embedding(seq=seq)
        for seq in data_test['siRNA_seq_for_mp_rna']
    ]
    sirna_embedding_df = pd.DataFrame(sirna_transformer_embedding, index=list(data_test['siRNA']))
    sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')].fillna(0) # Ensure no duplicates and fillna

    mrna_unique_seq_df = data_test.loc[:,['mRNA','mRNA_seq_for_mp_rna']].drop_duplicates(subset="mRNA")
    mrna_transformer_embedding = [
        utils.get_mp_rna_sequence_embedding(seq=seq)
        for seq in mrna_unique_seq_df['mRNA_seq_for_mp_rna']
    ]
    mrna_embedding_df = pd.DataFrame(mrna_transformer_embedding, index = list(mrna_unique_seq_df['mRNA']))
    mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')].fillna(0) # Ensure no duplicates and fillna
    print("MP-RNA Transformer embeddings generated.")


    # Positional encoding
    trans_table = str.maketrans('AUCG', 'UAGC') # Original was ATCG to TAGC, consistent with U in new data preparation
    # Use the original (T-based) or a consistent (U-based) sequence for this utility function,
    # as its behavior for 'T' vs 'U' might depend on the specific implementation of `find`.
    data_test['match_pos_siRNA_rev_comp'] = [seq[::-1].upper().translate(trans_table) for seq in data_test['siRNA_seq_for_other_features']]
    data_test['match_pos'] = data_test.apply(
        lambda row: row['mRNA_seq_for_other_features'].find(row['match_pos_siRNA_rev_comp'])
                    if row['mRNA_seq_for_other_features'].find(row['match_pos_siRNA_rev_comp']) != -1 else 0, # Handle not found cases
        axis=1
    )
    sirna_pos_encoding_per_interaction = [
        utils.get_pos_embedding_sequence(num, params["sirna_length"], params["dmodel"])
        for num in data_test['match_pos']
    ]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding_per_interaction, index = data_test['siRNA'] + '_' + data_test['mRNA']).fillna(0)
    sirna_pos_encoding = sirna_pos_encoding.loc[~sirna_pos_encoding.index.duplicated(keep='first')]


    # Thermodynamics (expects U)
    sirna_thermo_feat = [utils.cal_thermo_feature(seq) for seq in data_test['siRNA_seq_for_mp_rna']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    sirna_thermo_feat = pd.concat([data_test['siRNA'].reset_index(drop=True),
                                   data_test['mRNA'].reset_index(drop=True),
                                   sirna_thermo_feat],
                                   axis=1)
    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA']).fillna(0)


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
        sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0)
    except FileNotFoundError:
        print("Warning: siRNA self-fold feature file 'Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt' not found. Filling with zeros.")
        sirna_sfold_feat = pd.DataFrame(0.0, index=sirna_embedding_df.index, columns=[f'sfold_siRNA_dim_{i}' for i in range(6)])

    ## mRNA
    try:
        mrna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt", header=None, index_col=0)
        mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0)
    except FileNotFoundError:
        print("Warning: mRNA self-fold feature file 'Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt' not found. Filling with zeros.")
        mrna_sfold_feat = pd.DataFrame(0.0, index=mrna_embedding_df.index, columns=[f'sfold_mRNA_dim_{i}' for i in range(100)])


    # AGO2
    # siRNA-AGO2
    try:
        sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2_zh.csv", index_col=0)
        sirna_ago = sirna_ago.reindex(sirna_embedding_df.index).fillna(0)
    except FileNotFoundError:
        print("Warning: siRNA AGO2 feature file 'RNA_AGO2/siRNA_AGO2.csv' not found. Filling with zeros.")
        sirna_ago = pd.DataFrame(0.0, index=sirna_embedding_df.index, columns=[f'ago_siRNA_dim_0'])

    ## mRNA-AGO2
    try:
        mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2_zh.csv", index_col=0)
        mrna_ago = mrna_ago.reindex(mrna_embedding_df.index).fillna(0)
    except FileNotFoundError:
        print("Warning: mRNA AGO2 feature file 'RNA_AGO2/mRNA_AGO2.csv' not found. Filling with zeros.")
        mrna_ago = pd.DataFrame(0.0, index=mrna_embedding_df.index, columns=[f'ago_mRNA_dim_0'])


    # GC percentage
    ## siRNA (uses original `siRNA_seq`, `countGC` handles T/U internally)
    sirna_GC = [utils.countGC(seq) for seq in data_test['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC, index=list(data_test['siRNA'])).fillna(0)
    sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')]

    ## mRNA (uses original `mRNA_seq`, `countGC` handles T/U internally)
    mrna_unique_seq_df_orig = data_test.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA") # Use original seq for consistency
    mrna_GC = [utils.countGC(seq) for seq in mrna_unique_seq_df_orig['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_unique_seq_df_orig['mRNA'])).fillna(0)
    mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')]


    # k-mers (uses original `siRNA_seq`, kmer functions handle T/U internally)
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1).fillna(0)
    sirna_k_mers.index = data_test['siRNA']
    sirna_k_mers = sirna_k_mers.loc[~sirna_k_mers.index.duplicated(keep='first')]


    # siRNA rules codes (uses original `siRNA_seq`, `rules_scores` handles T/U internally)
    sirna_pos_scores = [utils.rules_scores(seq) for seq in data_test['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data_test['siRNA'])).fillna(0)
    sirna_pos_scores = sirna_pos_scores.loc[~sirna_pos_scores.index.duplicated(keep='first')]

    '''
    The features of GNN nodes
    '''

    # siRNA nodes
    # --- UPDATED: Use sirna_embedding_df instead of sirna_onehot ---
    sirna_pd = pd.concat([
        sirna_embedding_df,
        sirna_sfold_feat,
        sirna_ago,
        sirna_GC.reindex(sirna_embedding_df.index).fillna(0),
        sirna_k_mers.reindex(sirna_embedding_df.index).fillna(0),
        sirna_pos_scores.reindex(sirna_embedding_df.index).fillna(0)
    ], axis=1).fillna(0)


    # mRNA nodes
    # --- UPDATED: Use mrna_embedding_df instead of mrna_onehot ---
    mrna_pd = pd.concat([
        mrna_embedding_df,
        mrna_sfold_feat,
        mrna_ago,
        mrna_GC.reindex(mrna_embedding_df.index).fillna(0)
    ], axis=1).fillna(0)


    # interactive nodes (indexed by interaction ID)
    # Ensure all components have matching indices before concatenation
    interaction_pd = pd.concat([
        sirna_thermo_feat.reindex(sirna_thermo_feat.index).fillna(0), # Ensure this is primary index
        con_feat.reindex(sirna_thermo_feat.index).fillna(0),
        sirna_pos_encoding.reindex(sirna_thermo_feat.index).fillna(0)
    ], axis=1).fillna(0)

    #---------------------------------------------------

    # Create heterogeneous graph data
    data_hetero = HeteroData()

    # Add node features
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)

    # Create mapping from node names to indices (using the final feature DFs' indices)
    sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}

    # Create edge indices
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []

    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        # Ensure that the interaction actually exists in the processed interaction_pd index
        if interaction_name in interaction_idx and row['siRNA'] in sirna_idx and row['mRNA'] in mrna_idx:
            edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])
        else:
            print(f"Warning: Skipping edge for interaction {interaction_name} due to missing node features (e.g., from positional encoding failure).")


    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

    data_hetero = ToUndirected()(data_hetero)

    # Define split indices for NeighborLoader
    # These masks are typically used for in-graph message passing.
    # For NeighborLoader, we use input_nodes.
    train_mask = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.bool)
    dev_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    # Re-create interaction_idx based on the final interaction_pd for accurate mapping
    final_interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}

    # Assuming all entries in data_test are for the test set in this script
    test_indices_list = []
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in final_interaction_idx: # Only add if interaction node exists in graph
            test_indices_list.append(final_interaction_idx[interaction_name])
    test_idx = torch.tensor(test_indices_list, dtype=torch.long)


    test_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', test_idx),
        shuffle=False, # Crucial for consistent evaluation
        subgraph_type='induced',
        filter_per_worker=False
    )

    # Create labels tensor
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in final_interaction_idx: # Ensure interaction node exists in graph
            labels[final_interaction_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels


    # Initialize model
    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=params["dropout"]
    ).to(device)

    data_hetero = data_hetero.to(device) # Move the entire graph to the device

    # Load the saved weights
    model_path = f"model_fold{n}.pt" # <--- IMPORTANT: Ensure this path is correct for your saved models
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot test this fold.")
        continue # Skip to next fold
    except Exception as e:
        print(f"An error occurred while loading model for fold {n}: {e}. Skipping this fold.")
        continue

    # Set the model to evaluation mode
    model.eval()

    @torch.no_grad()
    def test(loader):
        model.eval()
        preds, truths = [], []

        for batch in loader:
            batch = batch.to(device) # Move batch data to the specified device

            batch_size = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size, device=device) # Ensure indices are on device too

            out = model(batch.x_dict, batch.edge_index_dict)

            # Ensure we only take predictions corresponding to the 'interaction' nodes in the batch
            preds.append(out[batch_indices].cpu())
            truths.append(batch['interaction'].y[batch_indices].cpu())

        return torch.cat(preds), torch.cat(truths)

    test_pred, test_true = test(test_loader)
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