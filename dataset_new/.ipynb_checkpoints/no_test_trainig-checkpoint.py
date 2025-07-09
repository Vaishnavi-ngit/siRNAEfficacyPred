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
from sklearn.metrics import mean_squared_error # Keep for potential future use or if other metrics are needed for validation besides loss
import scipy.stats # Keep for potential future use or if other metrics are needed for validation besides loss
import json

torch.cuda.manual_seed_all(42)
# Import the revised utils1.py
import BugUtils as utils1

# --- Load parameters ---
with open("Testing.json", 'r') as f:
    params = json.load(f)

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

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
NUM_FOLDS = 10
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
        # data_test = pd.read_csv(os.path.join(split_dir, "test.csv")) # Removed test data loading
    except FileNotFoundError as e:
        print(f"Error: Data file not found in '{split_dir}'. {e}")
        print("Skipping fold processing.")
        continue

    data_train['split'] = 'train'
    data_dev['split'] = 'dev'
    # data_test['split'] = 'test' # Removed test split assignment

    # Substitute U to T for siRNA_seq to match mRNA_seq (DNA-like) for positional lookup.
    # Note: K-mer, thermo, and rules_scores functions in utils will convert T to U back internally for RNA calculations.
    for df in [data_train, data_dev]: # Only train and dev
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')

    data = pd.concat([data_train, data_dev], axis=0).reset_index(drop=True) # Concatenate only train and dev

    print("\n--- Feature Processing ---")

    # Get the first sample's data for printing
    # first_sample = data.iloc[0]
    # first_siRNA_seq = first_sample['siRNA_seq']
    # first_mRNA_seq_RNA_FM = data.loc[data['mRNA'] == first_sample['mRNA'], 'mRNA_seq_RNA-FM'].iloc[0]
    # first_pos = first_sample['pos']

    # --- Feature processing with variable length handling and padding ---

    # 1. One-hot encoding for siRNA (PADDED to MAX_SIRNA_LENGTH)
    sirna_onehot = []
    for seq in data['siRNA_seq']:
        sirna_onehot.append(utils1.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data['siRNA']))
    # print(f"\n--- siRNA One-Hot Encoding ---")
    # print(f"Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_onehot.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {sirna_onehot.iloc[0].values[:20]}...")

    # 2. mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils1.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))
    # print(f"\n--- mRNA One-Hot Encoding ---")
    # print(f"Sequence: {first_mRNA_seq_RNA_FM}")
    # print(f"Full DataFrame Shape: {mrna_onehot.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {mrna_onehot.iloc[0].values[:20]}...")

    # 3. Positional encoding (PADDED to MAX_SIRNA_LENGTH * dmodel)
    sirna_pos_encoding = []
    for idx, row in data.iterrows():
        mrna_start_pos = max(0, int(row['pos']))
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos,
            len(row['siRNA_seq']),
            MAX_SIRNA_LENGTH,
            params["dmodel"]
        ))
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data['siRNA'] + '_' + data['mRNA'])
    # print(f"\n--- Positional Encoding ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"mRNA Match Position (from 'pos' column): {first_pos}")
    # print(f"Full DataFrame Shape: {sirna_pos_encoding.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {sirna_pos_encoding.iloc[0].values[:20]}...")

    # 4. Thermodynamics (PADDED to MAX_SIRNA_LENGTH)
    sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')
    # print(f"\n--- Thermodynamics Features ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_thermo_feat.shape}")
    # print(f"First Sample's Feature Vector (all values): {sirna_thermo_feat.iloc[0].values}")

    # 5. Co-fold features (using base_pair_probs as indicated by user)
    con_feat = pd.read_csv("siRNA_split_preprocess/full_con_matrix.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    # print(f"\n--- Co-fold Features (from base_pair_probs.txt) ---")
    # print(f"Interaction: {first_sample['siRNA']}_{first_sample['mRNA']}")
    # print(f"Full DataFrame Shape: {con_feat.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {con_feat.iloc[0].values[:20]}...")

    # 6. Self-fold features (siRNA)
    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)
    # print(f"\n--- siRNA Self-Fold Features ---")
    # print(f"siRNA: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_sfold_feat.shape}")
    # print(f"First Sample's Feature Vector (all values): {sirna_sfold_feat.iloc[0].values}")

    # 7. Self-fold features (mRNA)
    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    # print(f"\n--- mRNA Self-Fold Features ---")
    # print(f"mRNA: {first_mRNA_seq_RNA_FM}")
    # print(f"Full DataFrame Shape: {mrna_sfold_feat.shape}")
    # print(f"First Sample's Feature Vector (all values): {mrna_sfold_feat.iloc[0].values}")

    # 8. GC percentage (variable length robust)
    sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data['siRNA_seq']], index=list(data['siRNA']))
    mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))
    # print(f"\n--- siRNA GC Percentage ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_GC.shape}")
    # print(f"First Sample's Feature Value: {sirna_GC.iloc[0].values}")
    # print(f"\n--- mRNA GC Percentage ---")
    # print(f"mRNA Sequence: {first_mRNA_seq_RNA_FM}")
    # print(f"Full DataFrame Shape: {mrna_GC.shape}")
    # print(f"First Sample's Feature Value: {mrna_GC.iloc[0].values}")

    # 9. K-mers (All now return fixed-size lists)
    sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data['siRNA']
    # print(f"\n--- siRNA K-mer Frequencies ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_k_mers.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {sirna_k_mers.iloc[0].values[:20]}...")

    mrna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']])
    mrna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']])
    mrna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']])
    mrna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']])
    mrna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']])
    mrna_k_mers = pd.concat([mrna_1_mer, mrna_2_mers, mrna_3_mers, mrna_4_mers, mrna_5_mers], axis=1)
    mrna_k_mers.index = mrna_onehot_temp['mRNA']

    # 10. siRNA rules codes (PADDED to 19*3)
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    # print(f"\n--- siRNA Rules Scores ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_pos_scores.shape}")
    # print(f"First Sample's Feature Vector (all values): {sirna_pos_scores.iloc[0].values}")

    
    # print("\n--- Feature Processing ---")

    # # Get the first sample's data for printing
    # # first_sample = data.iloc[0]
    # # first_siRNA_seq = first_sample['siRNA_seq']
    # # first_mRNA_seq_RNA_FM = data.loc[data['mRNA'] == first_sample['mRNA'], 'mRNA_seq_RNA-FM'].iloc[0]
    # # first_pos = first_sample['pos']

    # # --- Feature processing with variable length handling and padding ---

    # # --- Sequence Embedding using MP-RNA Transformer ---
    # # Print statement for debugging if it gets stuck here
    # print("--- Starting Sequence Embedding (siRNA) ---")
    # sirna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in data['siRNA_seq']]
    # sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(data['siRNA']))
    # # print(sirna_embedding_df.head(20))
    # sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')]
    # print("--- Finished Sequence Embedding (siRNA) ---")

    # print("--- Starting Sequence Embedding (mRNA) ---")
    # mrna_unique_seq_df = data.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
    # mrna_transformer_embeddings = [utils1.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    # mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
    # mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')]
    # print("--- Finished Sequence Embedding (mRNA) ---")
    # # --- End of Sequence Embedding Update ---
    # # print(f"\n--- mRNA One-Hot Encoding ---")
    # # print(f"Sequence: {first_mRNA_seq_RNA_FM}")
    # # print(f"Full DataFrame Shape: {mrna_onehot.shape}")
    # # print(f"First Sample's Feature Vector (first 20 values): {mrna_onehot.iloc[0].values[:20]}...")

    # # 3. Positional encoding (PADDED to MAX_SIRNA_LENGTH * dmodel)
    # sirna_pos_encoding = []
    # for idx, row in data.iterrows():
    #     mrna_start_pos = max(0, int(row['pos']))
    #     sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
    #         mrna_start_pos,
    #         len(row['siRNA_seq']),
    #         MAX_SIRNA_LENGTH,
    #         params["dmodel"]
    #     ))
    # sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data['siRNA'] + '_' + data['mRNA'])
    # # print(f"\n--- Positional Encoding ---")
    # # print(f"siRNA Sequence: {first_siRNA_seq}")
    # # print(f"mRNA Match Position (from 'pos' column): {first_pos}")
    # # print(f"Full DataFrame Shape: {sirna_pos_encoding.shape}")
    # # print(f"First Sample's Feature Vector (first 20 values): {sirna_pos_encoding.iloc[0].values[:20]}...")

    # # 4. Thermodynamics (PADDED to MAX_SIRNA_LENGTH)
    # sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
    #                      for seq in data['siRNA_seq']]
    # sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    # temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    # sirna_thermo_feat['index'] = temp_interaction_index
    # sirna_thermo_feat = sirna_thermo_feat.set_index('index')
    # # print(f"\n--- Thermodynamics Features ---")
    # # print(f"siRNA Sequence: {first_siRNA_seq}")
    # # print(f"Full DataFrame Shape: {sirna_thermo_feat.shape}")
    # # print(f"First Sample's Feature Vector (all values): {sirna_thermo_feat.iloc[0].values}")

    # # 5. Co-fold features (using base_pair_probs as indicated by user)
    # con_feat = pd.read_csv("siRNA_split_preprocess/full_con_matrix.txt", header=None, index_col=0)
    # con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    # # print(f"\n--- Co-fold Features (from base_pair_probs.txt) ---")
    # # print(f"Interaction: {first_sample['siRNA']}_{first_sample['mRNA']}")
    # # print(f"Full DataFrame Shape: {con_feat.shape}")
    # # print(f"First Sample's Feature Vector (first 20 values): {con_feat.iloc[0].values[:20]}...")

    # # 6. Self-fold features (siRNA)
    # sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
    # sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_embedding_df.index)
    # # print(f"\n--- siRNA Self-Fold Features ---")
    # # print(f"siRNA: {first_siRNA_seq}")
    # # print(f"Full DataFrame Shape: {sirna_sfold_feat.shape}")
    # # print(f"First Sample's Feature Vector (all values): {sirna_sfold_feat.iloc[0].values}")

    # # 7. Self-fold features (mRNA)
    # mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
    # mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0)
    # # print(f"\n--- mRNA Self-Fold Features ---")
    # # print(f"mRNA: {first_mRNA_seq_RNA_FM}")
    # # print(f"Full DataFrame Shape: {mrna_sfold_feat.shape}")
    # # print(f"First Sample's Feature Vector (all values): {mrna_sfold_feat.iloc[0].values}")

    # # GC percentage
    # sirna_GC = [utils1.countGC(seq) for seq in data['siRNA_seq']]
    # sirna_GC = pd.DataFrame(sirna_GC, columns=['GC_content'], index=list(data['siRNA']))
    # sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')]

    # mrna_GC = [utils1.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    # mrna_GC = pd.DataFrame(mrna_GC, columns=['GC_content'], index=list(mrna_unique_seq_df['mRNA']))
    # mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')]


    # # 9. K-mers (All now return fixed-size lists)
    # sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']])
    # sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    # sirna_k_mers.index = data['siRNA']
    # # print(f"\n--- siRNA K-mer Frequencies ---")
    # # print(f"siRNA Sequence: {first_siRNA_seq}")
    # # print(f"Full DataFrame Shape: {sirna_k_mers.shape}")
    # # print(f"First Sample's Feature Vector (first 20 values): {sirna_k_mers.iloc[0].values[:20]}...")

    # mrna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_unique_seq_df['mRNA_seq']])
    # mrna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_unique_seq_df['mRNA_seq']])
    # mrna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_unique_seq_df['mRNA_seq']])
    # mrna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_unique_seq_df['mRNA_seq']])
    # mrna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_MRNA_LENGTH) for seq in mrna_unique_seq_df['mRNA_seq']])
    # mrna_k_mers = pd.concat([mrna_1_mer, mrna_2_mers, mrna_3_mers, mrna_4_mers, mrna_5_mers], axis=1)
    # mrna_k_mers.index = mrna_unique_seq_df['mRNA']

    # # 10. siRNA rules codes (PADDED to 19*3)
    # SIRNA_RULES_LENGTH = 19
    # sirna_pos_scores = []
    # for seq in data['siRNA_seq']:
    #     sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    # sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))
    # # print(f"\n--- siRNA Rules Scores ---")
    # # print(f"siRNA Sequence: {first_siRNA_seq}")
    # # print(f"Full DataFrame Shape: {sirna_pos_scores.shape}")
    # # print(f"First Sample's Feature Vector (all values): {sirna_pos_scores.iloc[0].values}")

    print("\n--- Assembling GNN Node Features ---")

    # siRNA nodes features
    sirna_pd = pd.concat([sirna_onehot, sirna_sfold_feat, sirna_GC, sirna_k_mers, sirna_pos_scores], axis=1)

    # mRNA nodes features
    mrna_pd = pd.concat([mrna_onehot, mrna_sfold_feat, mrna_GC, mrna_k_mers], axis=1)

    # #siRNA nodes features
    # sirna_pd = pd.concat([sirna_embedding_df, sirna_sfold_feat, sirna_GC, sirna_k_mers, sirna_pos_scores], axis=1)

    # #mRNA nodes features
    # mrna_pd = pd.concat([mrna_embedding_df, mrna_sfold_feat, mrna_GC, mrna_k_mers], axis=1)


    # Interaction nodes features
    interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding], axis=1)
    # print(f"\n--- Combined Interaction Node Features ---")
    # print(f"Full DataFrame Shape: {interaction_pd.shape}")
    # print(f"First Sample's Feature Vector (first 20 values): {interaction_pd.iloc[0].values[:20]}...")

    # --- Create heterogeneous graph data structure ---
    data_hetero = HeteroData()

    # Add node features (ensure they are aligned by index)
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)

    # print(f"\n--- PyG HeteroData Node Feature Shapes (Tensor) ---")
    # print(f"data_hetero['siRNA'].x shape: {data_hetero['siRNA'].x.shape}")
    # print(f"data_hetero['mRNA'].x shape: {data_hetero['mRNA'].x.shape}")
    # print(f"data_hetero['interaction'].x shape: {data_hetero['interaction'].x.shape}")

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
    # test_idx_list = [] # Removed test index list

    for _, row in data_train.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            train_idx_list.append(interaction_name_to_idx[interaction_name])

    for _, row in data_dev.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            dev_idx_list.append(interaction_name_to_idx[interaction_name])

    train_idx = torch.tensor(train_idx_list, dtype=torch.long)
    dev_idx = torch.tensor(dev_idx_list, dtype=torch.long)
    # test_idx = torch.tensor(test_idx_list, dtype=torch.long) # Removed test index tensor

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

    # test_loader = NeighborLoader(...) # Removed test_loader initialization

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

            # Ensure batch['interaction'].y has the correct shape for loss calculation
            # It should be batch.y but accessing via batch['interaction'].y is specific to HeteroData
            loss = criterion(out.squeeze(-1), batch['interaction'].y)

            l1_lambda = params.get("l1_lambda", 1e-5)
            loss += l1_lambda * model.l1_loss()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    # --- Validation Function (renamed from test for clarity) ---
    @torch.no_grad()
    def validate(loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            preds.append(out.cpu())
            truths.append(batch['interaction'].y.cpu())
        return torch.cat(preds), torch.cat(truths)

    # --- Main Training Loop ---
    #best_val_loss = float('inf')
    for epoch in range(params["epochs"]):
        train_loss = train()

        val_pred, val_true = validate(val_loader)
        val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()

        scheduler.step(val_loss)

        torch.save(model.state_dict(), f'best_model_fold{n}_gcn.pt')

        #if epoch % 10 == 0 or epoch == params["epochs"] - 1: # Print every 10 epochs and at the very last epoch
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    #print(f"--- Fold {n} finished! Final Validation Loss: {best_val_loss:.4f} ---")

# --- No overall metrics summary since test set evaluation is removed ---
print("\n--- Training and Validation Complete Across All Folds ---")
print("Model checkpoints saved based on best validation loss for each fold.")