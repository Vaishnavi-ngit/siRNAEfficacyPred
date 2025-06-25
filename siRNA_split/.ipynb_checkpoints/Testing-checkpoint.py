import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, LayerNorm, GATConv # Import GATConv and LayerNorm for potential future use or consistency
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
import utils  # Features calculation, assuming 'utils.py' is in the same directory or on PYTHONPATH

# Load parameters
# Ensure 'siRNA_param_pytorch.json' exists and contains necessary parameters
params = json.load(open("siRNA_param.json", 'r'))

# Initialize metric lists
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

# Model Definition: HeteroSAGE (as provided in the user's prompt for testing)
# Note: This model structure should match the one used during training
# for the saved 'best_model_fold{n}.pt' files to load correctly.
class HeteroSAGE(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0]  # (node_types, edge_types)

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
                if node_type in bn_dict:
                    x_dict[node_type] = bn_dict[node_type](x_dict[node_type])
                x_dict[node_type] = F.leaky_relu(x_dict[node_type])
                x_dict[node_type] = dropout(x_dict[node_type])
        return self.lin(x_dict['interaction'])

    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())

@torch.no_grad()
def evaluate_model(model, loader, device):
    """
    Evaluates the model on a given data loader.
    Args:
        model (torch.nn.Module): The GNN model to evaluate.
        loader (torch_geometric.loader.NeighborLoader): Data loader for evaluation.
        device (torch.device): The device (CPU/CUDA) to run the evaluation on.
    Returns:
        tuple: A tuple containing concatenated predictions and true labels.
    """
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch['interaction'].num_nodes
        batch_indices = torch.arange(batch_size, device=device) # Ensure indices are on the correct device

        out = model(batch.x_dict, batch.edge_index_dict)
        preds.append(out[batch_indices].cpu())
        truths.append(batch['interaction'].y[batch_indices].cpu())
    return torch.cat(preds), torch.cat(truths)


print(f"Parameters loaded: Dropout={params['dropout']}, Learning Rate={params['lr']}, Epochs={params['epochs']}")
print(f"HINSAGE Layer Sizes: {params['hinsage_layer_sizes']}, Hop Samples: {params['hop_samples']}")

# Loop through all 10 folds to test the model
for n in range(10):
    print(f"\n--- Testing on Fold {n} with new dataset ---")

    # Load the single CSV file for testing
    test_file = "siRNA_mRNA_all_data.csv"
    try:
        data_test = pd.read_csv(test_file)
        print(f"Successfully loaded test data from: {test_file}")
        print("First 10 rows of raw test data:\n", data_test.head(10))
    except FileNotFoundError:
        print(f"Error: File not found at {test_file}. Please check the file path.")
        continue # Continue to next fold instead of exiting
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        continue # Continue to next fold

    # Substitute U to T in siRNA sequence
    data_test['siRNA_seq'] = data_test['siRNA_seq'].str.replace('U', 'T')
    print("\n--- Test Data after 'U' to 'T' substitution (first 10 siRNA_seq) ---")
    print(data_test['siRNA_seq'].head(10))

    '''
    Feature processing
    '''
    print("\n--- Feature Processing ---")

    # one-hot encoding
    ## siRNA
    sirna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["sirna_length"]) for seq in data_test['siRNA_seq']]
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data_test['siRNA']))
    print("siRNA One-Hot Features (first 10 rows):\n", sirna_onehot.head(10))

    ## mRNA
    mrna_onehot_temp = data_test.loc[:, ['mRNA', 'mRNA_seq']]
    mrna_onehot_temp = mrna_onehot_temp.drop_duplicates(subset="mRNA")
    mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"]) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    print("mRNA One-Hot Features (first 10 rows):\n", mrna_onehot.head(10))


    # Positional encoding
    trans_table = str.maketrans('ATCG', 'TAGC')
    data_test['match_pos'] = [seq[::-1].upper().translate(trans_table) for seq in data_test['siRNA_seq']]
    data_test['match_pos'] = data_test.apply(lambda row: row['mRNA_seq'].index(row['match_pos']), axis=1)
    print("Test Data 'match_pos' (first 10 values):\n", data_test['match_pos'].head(10))

    sirna_pos_encoding = [utils.get_pos_embedding_sequence(num, params["sirna_length"], params["dmodel"]) for num in data_test['match_pos']]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=list(data_test['siRNA']))
    print("siRNA Positional Encoding Features (first 10 rows):\n", sirna_pos_encoding.head(10))


    # Thermodynamics
    sirna_thermo_feat = [utils.cal_thermo_feature(seq.replace("T", "U")) for seq in data_test['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    sirna_thermo_feat = pd.concat([data_test['siRNA'].reset_index(drop=True),
                                     data_test['mRNA'].reset_index(drop=True),
                                     sirna_thermo_feat],
                                     axis=1)
    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])
    print("siRNA Thermodynamics Features (first 10 rows):\n", sirna_thermo_feat.head(10))


    # Co-fold features
    # Ensure this path is correct for the new dataset's preprocessed files
    con_feat = pd.read_csv("Simone_split_preprocess/con_matrix_Simone_meanSum50.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index)
    print("Co-fold Features (first 10 rows):\n", con_feat.head(10))


    # sel-fold features
    ## siRNA
    # Ensure this path is correct for the new dataset's preprocessed files
    sirna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    print("siRNA Self-Fold Features (first 10 rows):\n", sirna_sfold_feat.head(10))

    ## mRNA
    # Ensure this path is correct for the new dataset's preprocessed files
    mrna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index)
    print("mRNA Self-Fold Features (first 10 rows):\n", mrna_sfold_feat.head(10))


    # AGO2
    ## siRNA-AGO2
    # Ensure this path is correct for the new dataset's preprocessed files
    sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2.csv", index_col=0)
    sirna_ago = sirna_ago.reindex(sirna_onehot.index)
    print("siRNA-AGO2 Features (first 10 rows):\n", sirna_ago.head(10))

    ## mRNA-AGO2
    # Ensure this path is correct for the new dataset's preprocessed files
    mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2.csv", index_col=0)
    mrna_ago = mrna_ago.reindex(mrna_onehot.index)
    print("mRNA-AGO2 Features (first 10 rows):\n", mrna_ago.head(10))


    # GC percentage
    ## siRNA
    sirna_GC = [utils.countGC(seq) for seq in data_test['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC, index=list(data_test['siRNA']))
    print("siRNA GC Percentage Features (first 10 rows):\n", sirna_GC.head(10))

    ## mRNA
    mrna_GC = [utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_onehot_temp['mRNA']))
    print("mRNA GC Percentage Features (first 10 rows):\n", mrna_GC.head(10))

    # k-mers
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data_test['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data_test['siRNA']
    print("siRNA K-mers Features (first 10 rows):\n", sirna_k_mers.head(10))


    # siRNA rules codes
    sirna_pos_scores = [utils.rules_scores(seq) for seq in data_test['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data_test['siRNA']))
    print("siRNA Rules Scores Features (first 10 rows):\n", sirna_pos_scores.head(10))
    
    '''
    The features of GNN nodes
    '''
    print("\n--- Combining Node Features for GNN ---")

    # siRNA nodes
    sirna_pd = pd.concat([sirna_onehot,sirna_sfold_feat,sirna_ago,sirna_GC,sirna_k_mers,sirna_pos_scores],axis = 1)
    print("Combined siRNA Node Features (first 10 rows):\n", sirna_pd.head(10))

    # mRNA nodes
    mrna_pd = pd.concat([mrna_onehot,mrna_sfold_feat,mrna_ago,mrna_GC],axis = 1)
    print("Combined mRNA Node Features (first 10 rows):\n", mrna_pd.head(10))

    # interactive nodes
    sirna_pos_encoding.index = sirna_thermo_feat.index # Re-align index for concatenation
    interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding], axis=1)
    print("Combined Interaction Node Features (first 10 rows):\n", interaction_pd.head(10))
    
    #---------------------------------------------------
    print("\n--- Creating Heterogeneous Graph Data ---")
    # Create heterogeneous graph data
    data_hetero = HeteroData()
    
    # Add node features
    # Ensure that all feature dataframes have consistent indices with the original data_test
    # This might require reindexing sirna_pd, mrna_pd, interaction_pd to data_test's original
    # siRNA, mRNA, and combined (siRNA_mRNA) indices if they were lost during feature computation.
    # The current code assumes indices align correctly from previous steps.

    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)
    
    print("HeteroData Node Features (Before Normalization):")
    print("siRNA.x[:10]:\n", data_hetero['siRNA'].x[:10])
    print("mRNA.x[:10]:\n", data_hetero['mRNA'].x[:10])
    print("interaction.x[:10]:\n", data_hetero['interaction'].x[:10])

    # Normalize features - CRUCIAL for consistency with training
    print("\n--- Normalizing HeteroData Node Features ---")
    for node_type in ['siRNA', 'mRNA', 'interaction']:
        x = data_hetero[node_type].x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-8
        data_hetero[node_type].x = (x - mean) / std
        print(f"{node_type}.x[:10] after normalization:\n", data_hetero[node_type].x[:10])

    # Add edges
    # Create mapping from node names to indices
    # Using unique values from the pandas DataFrames to ensure correct indexing
    sirna_unique_names = sirna_pd.index.unique().tolist()
    mrna_unique_names = mrna_pd.index.unique().tolist()
    interaction_unique_names = interaction_pd.index.unique().tolist() # Based on siRNA_mRNA_interaction_name

    sirna_idx = {name: i for i, name in enumerate(sirna_unique_names)}
    mrna_idx = {name: i for i, name in enumerate(mrna_unique_names)}
    interaction_idx = {name: i for i, name in enumerate(interaction_unique_names)}
    
    # Create edge indices
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []
    
    for _, row in data_test.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        # Ensure that the keys exist in the dictionaries before appending
        if row['siRNA'] in sirna_idx and row['mRNA'] in mrna_idx and interaction_name in interaction_idx:
            edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])
        else:
            # This indicates a mismatch between the interaction_pd.index and data_test rows.
            # This could happen if some interactions in data_test were not found in sirna_thermo_feat.index.
            # For debugging, you might want to print the missing keys.
            print(f"Warning: Missing index for interaction {interaction_name} or its components during edge creation.")
            
    
    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
        
    print("\n--- Initial Edge Indices (first 10 of each type) ---")
    print("siRNA-interaction edge_index[:, :10]:\n", data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index[:, :10])
    print("mRNA-interaction edge_index[:, :10]:\n", data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index[:, :10])

    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    
    print("\n--- Reversed Edge Indices (first 10 of each type) ---")
    print("interaction-siRNA edge_index[:, :10]:\n", data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index[:, :10])
    print("interaction-mRNA edge_index[:, :10]:\n", data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index[:, :10])

    # Apply ToUndirected transform
    data_hetero = ToUndirected()(data_hetero)
    print("\n--- HeteroData after ToUndirected transform ---")
    # This transform might add new edge types (e.g., ('interaction', 'interacts_with', 'siRNA'))
    # Print updated edge types to see what happened
    print("Updated data_hetero.edge_types:", data_hetero.edge_types)
    # Check one of the potentially new edges if they exist, e.g.,
    if ('interaction', 'interacts_with', 'siRNA') in data_hetero.edge_types:
        print("New interaction-siRNA edge_index[:, :10] (from ToUndirected):\n", data_hetero['interaction', 'interacts_with', 'siRNA'].edge_index[:, :10])

    # Create split indices for the current test data
    # Redefine interaction_idx based on the current 'data_test' to ensure it's accurate for test_idx
    interaction_idx_for_test = {f"{row.siRNA}_{row.mRNA}": i for i, row in enumerate(data_test.itertuples())}
    test_idx = torch.tensor([interaction_idx_for_test[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_test.iterrows() ], dtype=torch.long)
    print("\n--- Test Indices (first 10 values) ---")
    print("test_idx[:10]:", test_idx[:10])

    # Create labels tensor
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for i, row in data_test.iterrows(): # Iterate data_test to get corresponding efficacy for interactions
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_idx: # Use the original interaction_idx from feature engineering
            labels[interaction_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels
    print("\n--- Interaction Node Labels (efficacy) first 10 values ---")
    print("data_hetero['interaction'].y[:10]:\n", data_hetero['interaction'].y[:10])

    # Initialize model on the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recreate the model with same architecture as the one trained
    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=params["dropout"] # Use same dropout rate as training
    ).to(device)
    
    # It's good practice to move data_hetero to device if you're going to use it directly
    # in the model's forward pass outside of a loader, but NeighborLoader handles this for batches.
    # data_hetero = data_hetero.to(device) # Uncomment if you need the full graph on device

    # Load the saved weights for the current fold
    # Ensure 'best_model_fold{n}.pt' files are accessible from this script's execution environment
    model_path = f"best_model_fold{n}.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Skipping this fold.")
        continue # Skip this fold if model weights are not found
    except Exception as e:
        print(f"Error loading model weights for fold {n}: {e}. Skipping this fold.")
        continue

    # Set the model to evaluation mode
    model.eval()
        
    # Create NeighborLoader for testing
    # Note: test_idx should only contain indices of 'interaction' nodes that are actually in your test set
    test_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', test_idx),
        shuffle=False, # Must be False for consistent evaluation
        subgraph_type='induced',
        filter_per_worker=False
    )
    print("\n--- Running Model Evaluation ---")
    test_pred, test_true = evaluate_model(model, test_loader, device)

    test_pred = test_pred.cpu().numpy().flatten()
    test_true = test_true.cpu().numpy().flatten()
    
    # Filter out NaN values from predictions and true labels before calculating metrics
    valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
    if not np.any(valid_mask):
        print(f"Warning: All predictions or true values are NaN for fold {n}. Cannot calculate metrics.")
        continue
    
    test_pred_filtered = test_pred[valid_mask]
    test_true_filtered = test_true[valid_mask]

    print(f"\n--- Metrics for Fold {n} (filtered for NaN values) ---")
    # Calculate and print metrics
    try:
        r_value, _ = scipy.stats.pearsonr(test_true_filtered, test_pred_filtered)
        score_PCC.append(r_value)
        print("PCC:", r_value)
            
        spearman = scipy.stats.spearmanr(test_true_filtered, test_pred_filtered)
        score_SPCC.append(spearman[0])
        print("SPCC:", spearman[0])
            
        mse = mean_squared_error(test_true_filtered, test_pred_filtered)
        score_mse.append(mse)
        print("MSE:", mse)

        # Check for sufficient unique binary labels for AUC
        if len(np.unique(test_true_filtered > 0.7)) > 1:
            auc = roc_auc_score((test_true_filtered > 0.7).astype(int), test_pred_filtered)
            score_auc.append(auc)
            print("AUC:", auc)
        else:
            print("Warning: Not enough unique binary labels (test_true > 0.7) for AUC calculation in this fold.")

    except Exception as e:
        print(f"Error calculating metrics for fold {n}: {e}")
        continue
        
    print(f"Fold {n} finished!")

# Print final aggregated metrics
if score_PCC:
    print(f"\n--- Final Aggregated Results Across All Folds ({len(score_PCC)} successful folds) ---")
    print(f"Overall PCC score = {np.mean(score_PCC):.4f} ± {np.std(score_PCC):.4f}")
    print(f"Overall SPCC score = {np.mean(score_SPCC):.4f} ± {np.std(score_SPCC):.4f}")
    print(f"Overall MSE score = {np.mean(score_mse):.4f} ± {np.std(score_mse):.4f}")
    if score_auc:
        print(f"Overall AUC score = {np.mean(score_auc):.4f} ± {np.std(score_auc):.4f}")
    else:
        print("AUC could not be calculated for any fold or had insufficient data.")
else:
    print("\nNo successful folds were processed to calculate overall metrics.")
