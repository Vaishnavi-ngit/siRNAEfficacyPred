import os
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
import E_Utils as utils # Consistent import alias with training script

# --- Configuration ---
NEW_TEST_DATA_FILE = "siRNA_mRNA_all_data.csv" # !!! CHANGE THIS PATH to your new test dataset file !!!
# --- End Configuration ---

# Load parameters (should match parameters used during training from Exp.json)
try:
    params = json.load(open("siRNA_param_pytorch.json", 'r'))
except FileNotFoundError:
    print("Error: Exp.json not found. Please ensure it's in the same directory as this script.")
    exit()

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


print(f"Loading parameters from Exp.json...")
print(f"Dropout Rate: {params['dropout']}")
print(f"Learning Rate (from training params): {params['lr']}")
print(f"Epochs (from training params): {params['epochs']}")
print(f"HINSAGE Layer Sizes: {params['hinsage_layer_sizes']}")
print(f"Hop Samples: {params['hop_samples']}")

# Set device for PyTorch operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for GNN operations: {device}")

# Initialize metric lists to collect results for all folds
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

# Loop through folds (as in your training script for consistency)
for n in range(10): # Assuming folds are 0-9 based on the training script
    print(f"\n{'='*10} Testing Fold {n} {'='*10}")

    # Load the single CSV file for testing
    try:
        data_test = pd.read_csv(NEW_TEST_DATA_FILE)
        print(f"Successfully loaded test data from: {NEW_TEST_DATA_FILE}")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {NEW_TEST_DATA_FILE}. Skipping fold {n}.")
        continue
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}. Skipping fold {n}.")
        continue

    # --- Sequence Preprocessing (Identical to training script - T to U conversion for MP-RNA) ---
    data_test['siRNA_seq'] = data_test['siRNA_seq'].str.replace('T', 'U', regex=False)
    data_test['mRNA_seq'] = data_test['mRNA_seq'].str.replace('T', 'U', regex=False)

    # --- Feature processing (Identical to training script) ---
    print("--- Starting Feature Processing ---")

    # Sequence Embedding using MP-RNA Transformer
    print("  - Generating MP-RNA Transformer embeddings (siRNA)...")
    sirna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in data_test['siRNA_seq']]
    sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(data_test['siRNA']))
    sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')].fillna(0) # Ensure no duplicates and fillna

    print("  - Generating MP-RNA Transformer embeddings (mRNA)...")
    mrna_unique_seq_df = data_test.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
    mrna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
    mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')].fillna(0)

    # Positional encoding
    print("  - Calculating Positional Encoding...")
    trans_table = str.maketrans('AUCG', 'UAGC')
    data_test['match_pos_siRNA_rev_comp'] = [seq[::-1].upper().translate(trans_table) for seq in data_test['siRNA_seq']]
    data_test['match_pos'] = data_test.apply(
        lambda row: row['mRNA_seq'].find(row['match_pos_siRNA_rev_comp']),
        axis=1
    )
    # Use parameters with defaults for positional encoding (as in training)
    sirna_length = params.get("sirna_length", 19)
    dmodel = params.get("dmodel", 64)
    sirna_pos_encoding_per_interaction = [
        utils.get_pos_embedding_sequence(num, sirna_length, dmodel) for num in data_test['match_pos']
    ]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding_per_interaction, index = data_test['siRNA'] + '_' + data_test['mRNA'])
    sirna_pos_encoding = sirna_pos_encoding.loc[~sirna_pos_encoding.index.duplicated(keep='first')].fillna(0)


    # Thermodynamics
    print("  - Calculating Thermodynamics features...")
    sirna_thermo_feat_raw = [utils.cal_thermo_feature(seq) for seq in data_test['siRNA_seq']]
    sirna_thermo_feat_df = pd.DataFrame(sirna_thermo_feat_raw)

    sirna_thermo_feat = pd.concat([data_test['siRNA'].reset_index(drop=True),
                                   data_test['mRNA'].reset_index(drop=True),
                                   sirna_thermo_feat_df],
                                   axis = 1)
    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA']).fillna(0)

    # Co-fold features (using paths from training script's setup)
    print("  - Loading Co-fold features...")
    try:
        con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt", header=None, index_col=0)
        con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    except FileNotFoundError:
        print("    Warning: co-fold feature file 'siRNA_split_preprocess/con_matrix.txt' not found. Filling with zeros.")
        # Create a dummy DataFrame with appropriate columns if not found
        # Assuming a default dimension if the file is truly missing and cannot be inferred
        con_feat = pd.DataFrame(0.0, index=sirna_thermo_feat.index, columns=[f'cofold_feat_{i}' for i in range(50)]) # Placeholder columns


    # Self-fold features (using paths from training script's setup)
    print("  - Loading Self-fold features (siRNA)...")
    try:
        sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt", header=None, index_col=0)
        sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0)
    except FileNotFoundError:
        print("    Warning: siRNA self-fold feature file 'siRNA_split_preprocess/self_siRNA_matrix.txt' not found. Filling with zeros.")
        sirna_sfold_feat = pd.DataFrame(0.0, index=sirna_embedding_df.index, columns=[f'sfold_siRNA_feat_{i}' for i in range(6)]) # Placeholder columns

    print("  - Loading Self-fold features (mRNA)...")
    try:
        mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0)
        mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0)
    except FileNotFoundError:
        print("    Warning: mRNA self-fold feature file 'siRNA_split_preprocess/self_mRNA_matrix.txt' not found. Filling with zeros.")
        mrna_sfold_feat = pd.DataFrame(0.0, index=mrna_embedding_df.index, columns=[f'sfold_mRNA_feat_{i}' for i in range(100)]) # Placeholder columns

    # GC percentage
    print("  - Calculating GC percentage...")
    sirna_GC = [utils.countGC(seq) for seq in data_test['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC, columns=['GC_content'], index=list(data_test['siRNA']))
    sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')].fillna(0)

    mrna_GC = [utils.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, columns=['GC_content'], index=list(mrna_unique_seq_df['mRNA']))
    mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')].fillna(0)

    # AU percentage (NEW FEATURE)
    print("  - Calculating AU Content...")
    sirna_AU = [utils.countAU(seq) for seq in data_test['siRNA_seq']]
    sirna_AU = pd.DataFrame(sirna_AU, columns=['AU_content'], index=list(data_test['siRNA']))
    sirna_AU = sirna_AU.loc[~sirna_AU.index.duplicated(keep='first')].fillna(0)

    mrna_AU = [utils.countAU(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_AU = pd.DataFrame(mrna_AU, columns=['AU_content'], index=list(mrna_unique_seq_df['mRNA']))
    mrna_AU = mrna_AU.loc[~mrna_AU.index.duplicated(keep='first')].fillna(0)

    # Regional GC Content (NEW FEATURE)
    print("  - Calculating Regional GC Content...")
    regional_gc_window = params.get("regional_gc_window", 5)
    regional_gc_overlap = params.get("regional_gc_overlap", 0)
    sirna_regional_GC_raw = [utils.get_regional_gc_content(seq, window_size=regional_gc_window, overlap=regional_gc_overlap) for seq in data_test['siRNA_seq']]
    num_regional_gc_features = max((len(features) for features in sirna_regional_GC_raw), default=0)
    sirna_regional_GC_df = pd.DataFrame(sirna_regional_GC_raw, index=list(data_test['siRNA']),
                                        columns=[f'regional_GC_{i}' for i in range(num_regional_gc_features)])
    sirna_regional_GC_df = sirna_regional_GC_df.loc[~sirna_regional_GC_df.index.duplicated(keep='first')].fillna(0)

    # Terminal Base One-Hot Encoding (NEW FEATURE)
    print("  - Calculating Terminal Base One-Hot Encoding...")
    sirna_terminal_onehot_raw = [utils.get_terminal_base_one_hot(seq, length=sirna_length) for seq in data_test['siRNA_seq']]
    sirna_terminal_onehot = pd.DataFrame(sirna_terminal_onehot_raw, index=list(data_test['siRNA']))
    sirna_terminal_onehot.columns = ['5_prime_A', '5_prime_C', '5_prime_G', '5_prime_U',
                                      '3_prime_A', '3_prime_C', '3_prime_G', '3_prime_U']
    sirna_terminal_onehot = sirna_terminal_onehot.loc[~sirna_terminal_onehot.index.duplicated(keep='first')].fillna(0)

    # Thermodynamic Asymmetry (NEW FEATURE)
    print("  - Calculating Thermodynamic Asymmetry...")
    asym_5_prime_len = params.get("asym_5_prime_len", 5)
    asym_3_prime_len = params.get("asym_3_prime_len", 5)
    sirna_thermo_asymmetry_raw = [utils.get_thermo_asymmetry(seq, region1_len=asym_5_prime_len, region2_len=asym_3_prime_len) for seq in data_test['siRNA_seq']]
    sirna_thermo_asymmetry = pd.DataFrame(sirna_thermo_asymmetry_raw, columns=['thermo_asymmetry'], index=data_test['siRNA'] + '_' + data_test['mRNA'])
    sirna_thermo_asymmetry = sirna_thermo_asymmetry.loc[~sirna_thermo_asymmetry.index.duplicated(keep='first')].fillna(0)

    # k-mers
    print("  - Calculating k-mers...")
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data_test['siRNA_seq']]).fillna(0)
    sirna_k_mers = pd.concat([sirna_1_mer,sirna_2_mers, sirna_3_mers,sirna_4_mers,sirna_5_mers], axis = 1).fillna(0)
    sirna_k_mers.index = data_test['siRNA']
    sirna_k_mers = sirna_k_mers.loc[~sirna_k_mers.index.duplicated(keep='first')]

    # siRNA rules codes
    print("  - Calculating siRNA rules scores...")
    sirna_pos_scores = [utils.rules_scores(seq) for seq in data_test['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index = list(data_test['siRNA'])).fillna(0)
    sirna_pos_scores = sirna_pos_scores.loc[~sirna_pos_scores.index.duplicated(keep='first')]


    # The features of GNN nodes (NOW INCLUDING NEW FEATURES, Identical to training script)
    print("--- Consolidating Node Features ---")
    print("  - siRNA nodes...")
    sirna_pd = pd.concat([
        sirna_embedding_df,
        sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0), # Reindex to align with embedding
        sirna_GC.reindex(sirna_embedding_df.index).fillna(0),
        sirna_AU.reindex(sirna_embedding_df.index).fillna(0),
        sirna_regional_GC_df.reindex(sirna_embedding_df.index).fillna(0),
        sirna_terminal_onehot.reindex(sirna_embedding_df.index).fillna(0),
        sirna_k_mers.reindex(sirna_embedding_df.index).fillna(0),
        sirna_pos_scores.reindex(sirna_embedding_df.index).fillna(0)
    ], axis = 1).fillna(0)
    print(f"    siRNA_pd shape: {sirna_pd.shape}")


    print("  - mRNA nodes...")
    mrna_pd = pd.concat([
        mrna_embedding_df,
        mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0), # Reindex to align with embedding
        mrna_GC.reindex(mrna_embedding_df.index).fillna(0),
        mrna_AU.reindex(mrna_embedding_df.index).fillna(0)
    ], axis = 1).fillna(0)
    print(f"    mRNA_pd shape: {mrna_pd.shape}")


    print("  - Interaction nodes...")
    interaction_pd = pd.concat([
        sirna_thermo_feat, # This is the primary index for interaction_pd
        con_feat.reindex(sirna_thermo_feat.index).fillna(0),
        sirna_pos_encoding.reindex(sirna_thermo_feat.index).fillna(0),
        sirna_thermo_asymmetry.reindex(sirna_thermo_feat.index).fillna(0)
    ],axis=1).fillna(0)
    print(f"    interaction_pd shape: {interaction_pd.shape}")
    print("--- Feature Processing Complete ---")

    # --- Create heterogeneous graph data (Identical to training script) ---
    print("--- Creating HeteroData object ---")
    data_hetero = HeteroData()

    # Add node features
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)
    print("  Node features added.")

    # Create mapping from node names to indices (using the final feature DFs' indices)
    sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}
    print("  Node index mappings created.")

    # Create edge indices
    print("  Creating Edge Indices...")
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []

    # Iterate over the original 'data_test' to build edges, ensuring nodes exist in final DFs
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        # Ensure that the interaction actually exists in the processed interaction_pd index
        if (interaction_name in interaction_idx and
            row['siRNA'] in sirna_idx and
            row['mRNA'] in mrna_idx):
            edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])
        # else:
        #     # This warning can be useful for debugging if interactions are unexpectedly filtered out
        #     print(f"    Warning: Skipping edge for interaction {interaction_name} due to missing node features.")


    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    print("  Edge indices tensors created.")

    data_hetero = ToUndirected()(data_hetero)
    print("  Graph converted to undirected.")


    # Create test indices for the NeighborLoader
    # Re-create final_interaction_idx based on the final interaction_pd for accurate mapping
    final_interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}

    test_indices_list = []
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in final_interaction_idx: # Only add if interaction node exists in graph
            test_indices_list.append(final_interaction_idx[interaction_name])
    test_idx = torch.tensor(test_indices_list, dtype=torch.long)
    print(f"  Test indices created ({len(test_idx)} interactions).")

    test_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', test_idx),
        shuffle=False, # Crucial for consistent evaluation
        subgraph_type='induced',
        filter_per_worker=False
    )
    print("  NeighborLoader created for testing.")

    # Create labels tensor
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in final_interaction_idx: # Ensure interaction node exists in graph
            labels[final_interaction_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels
    print("  Labels assigned to graph.")
    print("--- HeteroData Object Creation Complete ---")


    # Initialize model
    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=params["dropout"]
    ).to(device)

    # Move the entire graph to the device (for NeighborLoader)
    data_hetero = data_hetero.to(device)

    # Load the saved weights
    model_path = f"model_fold{0}.pt" # Corrected path to match saving in training script
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot test fold {n}. Skipping to next fold.")
        continue # Skip to next fold
    except Exception as e:
        print(f"An error occurred while loading model for fold {n}: {e}. Skipping this fold.")
        continue

    # Set the model to evaluation mode
    model.eval()

    @torch.no_grad()
    def test_inference(loader):
        model.eval() # Ensure model is in eval mode (redundant but safe)
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

    test_pred, test_true = test_inference(test_loader)
    test_pred = test_pred.numpy().flatten()
    test_true = test_true.numpy().flatten()

    # Filter out NaNs if any
    valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
    if not np.any(valid_mask):
        print(f"Warning: All predictions or true values are NaN for fold {n}. Skipping metrics.")
        continue

    test_pred = test_pred[valid_mask]
    test_true = test_true[valid_mask]

    # Calculate metrics
    print("\n--- Test Metrics for this Fold ---")
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
    if len(np.unique(test_true > 0.7)) > 1: # Check if there are at least two unique binary classes
        auc = roc_auc_score((test_true > 0.7).astype(int), test_pred)
        print(f"AUC: {auc:.4f}")
        score_auc.append(auc)
    else:
        print(f"AUC: Not enough unique classes for AUC calculation (all 'efficacy' values are on one side of 0.7 threshold).")

    print(f"Fold {n} finished!")


### Overall Cross-Validation Metrics:

if score_PCC: # Only calculate if at least one fold was successfully processed
    print(f"\n{'='*40}\nOverall Cross-Validation Metrics:")
    print(f"Overall PCC score = {np.mean(score_PCC):.4f} +/- {np.std(score_PCC):.4f}")
    print(f"Overall SPCC score = {np.mean(score_SPCC):.4f} +/- {np.std(score_SPCC):.4f}")
    print(f"Overall MSE score = {np.mean(score_mse):.4f} +/- {np.std(score_mse):.4f}")
    if score_auc:
        print(f"Overall AUC score = {np.mean(score_auc):.4f} +/- {np.std(score_auc):.4f}")
    else:
        print("Overall AUC score not calculated (not enough true classes for AUC across all successful folds).")
else:
    print("\nNo folds were successfully tested. Please check the data paths and model paths.")