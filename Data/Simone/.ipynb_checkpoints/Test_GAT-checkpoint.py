import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" # Adjust as needed for your testing environment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GraphConv, Linear
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import scipy.stats
import json

# Make sure utils1.py is in the same directory or accessible via PYTHONPATH
import utils1

# --- Configuration for Testing ---
# IMPORTANT: Adjust these paths for your specific test dataset and trained model location
TEST_DATA_CSV = "siRNA_mRNA_all_data.csv" # Path to your new test CSV file

# Specify which trained model to load (e.g., from fold 0).
# Ensure this matches the name of your saved model file.
MODEL_TO_TEST_PATH = 'best_model_fold0.pt'

# Ensure these preprocessing files exist and are correctly located relative to where you run the script
# THESE ARE THE ORIGINAL PATHS FROM YOUR CODE - CONFIRMED.
FULL_CON_MATRIX_PATH = "Simone_split_preprocess/con_matrix_Simone_meanSum50.txt"
FULL_SELF_SIRNA_MATRIX_PATH = "Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt"
FULL_SELF_MRNA_MATRIX_PATH = "Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt"

# --- Load parameters ---
try:
    # Changed back to Testing.json as per your original code
    with open("Testing.json", 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print("Error: Testing.json not found. Make sure it's in the same directory.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from Testing.json: {e}")
    print("Please check for syntax errors like missing commas or unquoted keys/values.")
    exit()


MAX_SIRNA_LENGTH = params["sirna_length"]

# --- HIN-SAGE Model Class (Identical to your training script) ---
class HeteroSAGE(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0] # (node_types, edge_types)

        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else -1 # -1 lets GraphConv infer
            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): GraphConv((in_channels, in_channels), out_channels_i),
                ('mRNA', 'interacts_with', 'interaction'): GraphConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'siRNA'): GraphConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'mRNA'): GraphConv((in_channels, in_channels), out_channels_i),
            }, aggr='mean')
            self.convs.append(conv)

            # BatchNorm for each node type
            self.bns.append(nn.ModuleDict({
                node_type: BatchNorm1d(out_channels_i) for node_type in self.node_types
            }))

            self.dropouts.append(Dropout(dropout_rate))

        self.lin = Linear(layer_sizes[-1], out_channels)

        # --- Consistency: Initialize weights (even if not strictly needed for inference) ---
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_dict, edge_index_dict):
        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                # Apply BatchNorm only if the node_type exists in the current bn_dict and batch size > 1 or in eval mode
                if node_type in bn_dict and x_dict[node_type].size(0) > 0: # Added x_dict[node_type].size(0) > 0 check
                    if x_dict[node_type].size(0) > 1 or not self.training:
                        x_dict[node_type] = bn_dict[node_type](x_dict[node_type])
                x_dict[node_type] = F.leaky_relu(x_dict[node_type])
                # Dropout should be off in eval due to model.eval() being set outside,
                # but the layer itself is still part of the architecture.
                x_dict[node_type] = dropout(x_dict[node_type])
        return self.lin(x_dict['interaction'])

    def l1_loss(self):
        # L1 loss is typically for training, but kept for model definition consistency
        return sum(p.abs().sum() for p in self.parameters())

print(f"Loaded Parameters: {params}")
print(f"Max siRNA Length for Padding: {MAX_SIRNA_LENGTH}")

# Set device for PyTorch operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for GNN operations: {device}")


def run_test():
    print(f"\n--- Starting Testing with Model: {MODEL_TO_TEST_PATH} ---")

    # Load the single CSV file for testing
    try:
        data_test = pd.read_csv(TEST_DATA_CSV)
        print(f"Successfully loaded test data from: {TEST_DATA_CSV}")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {TEST_DATA_CSV}. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}.")
        return

    # Prepare sequences for feature calculation
    # Apply the same sequence preprocessing as in training
    data_test['siRNA_seq'] = data_test['siRNA_seq'].str.replace('U', 'T', regex=False)
    data_test['mRNA_seq'] = data_test['mRNA_seq'].str.replace('U', 'T', regex=False)

    print("\n--- Feature Processing for Test Data (Matching Training) ---")

    # --- Pre-loading global feature matrices (matching training setup) ---
    print("\n--- Pre-loading global feature matrices for testing ---")
    try:
        global_con_feat = pd.read_csv(FULL_CON_MATRIX_PATH, header=None, index_col=0)
        global_sirna_sfold_feat = pd.read_csv(FULL_SELF_SIRNA_MATRIX_PATH, header=None, index_col=0)
        global_mrna_sfold_feat = pd.read_csv(FULL_SELF_MRNA_MATRIX_PATH, header=None, index_col=0)
        print("Global feature pre-loading successful.")
    except FileNotFoundError as e:
        print(f"Error loading global feature matrix: {e}. Please ensure files are in 'Simone_split_preprocess/'.")
        print("Attempting to proceed by creating empty feature dataframes, but this may lead to errors or poor performance.")
        # Create empty dataframes with dummy columns to allow concatenation later if files are missing
        # This is a fallback and might not work if dimensions don't match trained model's expectations.
        global_con_feat = pd.DataFrame(0.0, index=[], columns=[f'col_{i}' for i in range(50)]) # Assuming 50 dimensions from original code
        global_sirna_sfold_feat = pd.DataFrame(0.0, index=[], columns=[f'col_{i}' for i in range(6)]) # Assuming 6 dimensions
        global_mrna_sfold_feat = pd.DataFrame(0.0, index=[], columns=[f'col_{i}' for i in range(100)]) # Assuming 100 dimensions


    print("--- Starting Sequence Embedding (siRNA) ---")
    siRNA_unique_seq_df = data_test.loc[:, ['siRNA', 'siRNA_seq']].drop_duplicates(subset="siRNA").copy()
    sirna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in siRNA_unique_seq_df['siRNA_seq']]
    sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(siRNA_unique_seq_df['siRNA']))
    sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')].copy() # Added .copy()
    print("--- Finished Sequence Embedding (siRNA) ---")

    print("--- Starting Sequence Embedding (mRNA) ---")
    mrna_unique_seq_df = data_test.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA").copy()
    mrna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
    mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')].copy() # Added .copy()
    print("--- Finished Sequence Embedding (mRNA) ---")
    
    # Positional encoding
    sirna_pos_encoding = []
    for _, row in data_test.iterrows():
        mrna_start_pos = max(0, int(row['pos']))
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos,
            len(row['siRNA_seq']),
            MAX_SIRNA_LENGTH,
            params["dmodel"]
        ))
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data_test['siRNA'] + '_' + data_test['mRNA']).copy() # Added .copy()
    
    # Thermodynamics
    sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data_test['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    temp_interaction_index = data_test['siRNA'] + '_' + data_test['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').copy() # Added .copy()

    # Co-fold features (reindexed from pre-loaded global data)
    con_feat = global_con_feat.reindex(sirna_thermo_feat.index).fillna(0).copy() # Added .copy()

    # Self-fold features siRNA (reindexed from pre-loaded global data)
    sirna_sfold_feat = global_sirna_sfold_feat.reindex(siRNA_unique_seq_df['siRNA']).fillna(0).copy() # Added .copy()

    # Self-fold features mRNA (reindexed from pre-loaded global data)
    mrna_sfold_feat = global_mrna_sfold_feat.reindex(mrna_unique_seq_df['mRNA']).fillna(0).copy() # Added .copy()

    # GC percentage
    sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data_test['siRNA_seq']], columns=['GC_content'], index=list(data_test['siRNA']))
    sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')].copy() # Added .copy()

    mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']], columns=['GC_content'], index=list(mrna_unique_seq_df['mRNA']))
    mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')].copy() # Added .copy()

    # K-mers
    sirna_1_mer = pd.DataFrame([utils1.single_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils1.double_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data_test['siRNA']
    sirna_k_mers = sirna_k_mers.loc[~sirna_k_mers.index.duplicated(keep='first')].copy() # Added .copy()

    # siRNA rules scores
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data_test['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data_test['siRNA']))
    sirna_pos_scores = sirna_pos_scores.loc[~sirna_pos_scores.index.duplicated(keep='first')].copy() # Added .copy()


    print("\n--- Assembling GNN Node Features (Matching Training) ---")

    # siRNA nodes features
    # Ensure all components have matching indices or are handled by fillna(0)
    sirna_pd = pd.concat([
        sirna_embedding_df,
        sirna_sfold_feat,
        sirna_GC,
        sirna_k_mers,
        sirna_pos_scores
    ], axis=1).fillna(0).copy() # Final fillna(0) and copy

    # mRNA nodes features
    mrna_pd = pd.concat([
        mrna_embedding_df,
        mrna_sfold_feat,
        mrna_GC
    ], axis=1).fillna(0).copy() # Final fillna(0) and copy

    # interactive nodes (indexed by interaction ID)
    interaction_pd = pd.concat([
        sirna_thermo_feat,
        con_feat,
        sirna_pos_encoding
    ], axis=1).fillna(0).copy() # Final fillna(0) and copy

    # Create heterogeneous graph data
    data_hetero = HeteroData()

    # Add node features
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)

    # Create mapping from node names to indices (using the final feature DFs' indices)
    sirna_name_to_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_name_to_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_name_to_idx = {name: i for i, name in enumerate(interaction_pd.index)}

    # Create edge indices and collect original efficacies for metric calculation
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []
    # original_efficacies_for_metrics is not used if labels are directly from data_hetero['interaction'].y
    # We will derive the true labels from data_hetero['interaction'].y later based on test_idx

    for _, row in data_test.iterrows():
        siRNA_name = row['siRNA']
        mRNA_name = row['mRNA']
        interaction_name = f"{siRNA_name}_{mRNA_name}"

        # Ensure that the interaction and its constituent nodes exist in the processed feature dataframes
        if (interaction_name in interaction_name_to_idx and
            siRNA_name in sirna_name_to_idx and
            mRNA_name in mrna_name_to_idx):
            
            edge_index_siRNA_to_interaction.append([sirna_name_to_idx[siRNA_name], interaction_name_to_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_name_to_idx[mRNA_name], interaction_name_to_idx[interaction_name]])
        else:
            print(f"Warning: Skipping edge for interaction {interaction_name} due to missing node features. This interaction will not be included in testing.")

    if not edge_index_siRNA_to_interaction or not edge_index_mRNA_to_interaction:
        print("Error: No valid interactions found to build the graph and conduct testing. Check data and feature processing.")
        return

    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

    # Automatically add reverse edges and make graph undirected
    data_hetero = ToUndirected()(data_hetero)

    # Create labels tensor for ALL 'interaction' nodes in the data_hetero graph.
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    # Populate labels based on the data_test rows that map to graph interactions
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            labels[interaction_name_to_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels


    # Define test indices for NeighborLoader
    # Ensure test_idx only contains indices of interactions that were successfully added to the graph
    test_indices_list = [interaction_name_to_idx[f"{row['siRNA']}_{row['mRNA']}"]
                         for _, row in data_test.iterrows()
                         if f"{row['siRNA']}_{row['mRNA']}" in interaction_name_to_idx]

    if not test_indices_list:
        print("Error: No valid interactions to test after graph construction. Check your test data and feature files.")
        return
        
    test_idx = torch.tensor(test_indices_list, dtype=torch.long)
    print(f"Number of test interactions in graph: {len(test_idx)}")


    test_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', test_idx), # Only load relevant test interactions
        shuffle=False, # Crucial for consistent evaluation
        subgraph_type='induced',
        filter_per_worker=params.get("filter_per_worker", True), # Match training setting
        num_workers=params.get("num_workers", 0) # Match training setting
    )

    # Initialize model
    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=0.0 # Set dropout to 0 for inference
    ).to(device)

    # Load the saved weights
    try:
        model.load_state_dict(torch.load(MODEL_TO_TEST_PATH, map_location=device))
        print(f"Successfully loaded model weights from: {MODEL_TO_TEST_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_TO_TEST_PATH}. Please ensure the trained model exists.")
        return
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print("This might happen if the model architecture or feature dimensions have changed since training.")
        print("Please ensure your 'Testing.json' and feature data match what was used for training.")
        return

    # Set the model to evaluation mode
    model.eval()

    @torch.no_grad()
    def evaluate_model(loader):
        model.eval() # Ensure model is in eval mode
        preds, truths = [], []

        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            
            # The output of NeighborLoader for the target node type ('interaction')
            # is typically the first batch_size elements of the full output.
            # `batch['interaction'].batch_size` gives the number of original `input_nodes` in the batch.
            preds.append(out[:batch['interaction'].batch_size].cpu().squeeze(-1))
            truths.append(batch['interaction'].y[:batch['interaction'].batch_size].cpu())

        return torch.cat(preds), torch.cat(truths)

    test_pred, test_true = evaluate_model(test_loader)
    
    # Ensure they are numpy arrays and flat
    test_pred = test_pred.numpy().flatten()
    test_true = test_true.numpy().flatten()

    # Filter out NaNs if any (should be less likely due to fillna, but good practice)
    valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
    if not np.any(valid_mask):
        print("Warning: All predictions or true values are NaN for the test set. Skipping metrics calculation.")
        return

    test_pred = test_pred[valid_mask]
    test_true = test_true[valid_mask]

    if len(test_pred) == 0:
        print("Error: No valid predictions/true values after filtering NaNs. Cannot compute metrics.")
        return

    # --- Calculate Metrics ---
    print(f"\n--- Test Set Metrics for {MODEL_TO_TEST_PATH} ---")

    # 1. Pearson Correlation Coefficient (PCC)
    pcc, _ = scipy.stats.pearsonr(test_true, test_pred)
    print(f"Pearson Correlation Coefficient (PCC): {pcc:.4f}")

    # 2. Spearman's Rank Correlation Coefficient (SPCC)
    spearman_rho, _ = scipy.stats.spearmanr(test_true, test_pred)
    print(f"Spearman's Rho (SPCC): {spearman_rho:.4f}")

    # 3. Mean Squared Error (MSE)
    mse = mean_squared_error(test_true, test_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    # 4. R-squared (R2) Score (Added for regression evaluation)
    r2 = r2_score(test_true, test_pred)
    print(f"R-squared (R2) Score: {r2:.4f}")

    # 5. Area Under ROC Curve (AUC)
    # Define a threshold for converting continuous efficacy to a binary label.
    # You can adjust this threshold in your Testing.json if needed.
    efficacy_threshold_for_auc = params.get("efficacy_binary_threshold", 0.5) # Default to 0.5 if not in params
    
    # Convert continuous true efficacy to binary true labels for AUC
    y_true_binary_auc = (test_true > efficacy_threshold_for_auc).astype(int)

    # Use the predicted efficacy values directly as scores for AUC.
    # If your model was trained for regression, these are not true probabilities,
    # but AUC can still give insight into ranking ability.
    y_score_for_auc = test_pred

    # Check if there are at least two unique classes for AUC calculation
    if len(np.unique(y_true_binary_auc)) < 2:
        print(f"AUC (Area Under ROC Curve): Cannot compute AUC. True binary labels (based on threshold {efficacy_threshold_for_auc}) have fewer than 2 unique classes.")
    else:
        try:
            auc = roc_auc_score(y_true_binary_auc, y_score_for_auc)
            print(f"AUC (Area Under ROC Curve): {auc:.4f}")
        except ValueError as e:
            print(f"Error computing AUC: {e}")
            print("Ensure y_true_binary_auc contains at least two classes and y_score_for_auc are suitable scores.")


    print("\n--- Testing Complete ---")

# --- Run the Test ---
if __name__ == "__main__":
    run_test()