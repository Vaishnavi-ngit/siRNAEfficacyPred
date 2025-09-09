import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, GraphConv
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
from torch.cuda.amp import GradScaler, autocast
import largeDataUtils as utils1 # Make sure utils.py is in the same directory or accessible via PYTHONPATH

# --- Load parameters ---
with open("siRNA_param_pytorch.json", 'r') as f:
    params = json.load(f)


# Initialize metric lists
score_PCC = []
score_SPCC = []
score_mse = []
score_auc = []

# --- Define Token IDs as constants for consistency ---
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 5 # Used for CLS token

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

# Add default values for missing parameters
params.setdefault("embedding_dim", 32)  # Default embedding dimension
params.setdefault("n_head", 4)           # Default number of attention heads
params.setdefault("n_layers", 2)         # Default number of transformer layers
# Ensure dmodel is set for positional encoding in utils1

# Define device here, at the top level, so it's accessible globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenize_sequence(seq, max_len):
    seq = seq.replace('U', 'T').upper()
    
    token_map = {
        'A': 1, 'T': 2, 'G': 3, 'C': 4,
    }

    sequence_tokens = [token_map.get(base, PAD_TOKEN_ID) for base in seq]

    # Prepend the CLS token
    tokens_with_cls = [CLS_TOKEN_ID] + sequence_tokens

    # Handle padding and truncation
    if len(tokens_with_cls) < max_len:
        tokens_with_cls += [PAD_TOKEN_ID] * (max_len - len(tokens_with_cls))
    else:
        tokens_with_cls = tokens_with_cls[:max_len]
        
    return tokens_with_cls

# --- TransformerEncoder Class ---
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_head, n_layers, dropout = 0.3):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID) 
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model) * 0.1)

        encoder_layers = []
        for _ in range(n_layers):
            encoder_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ))
        self.transformer_encoder = nn.Sequential(*encoder_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2 , d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x_tokens, additional_features=None):
        seq_len = x_tokens.size(1)
        src = self.embedding(x_tokens) * math.sqrt(self.d_model)
        src = src + self.pos_encoding[:seq_len].unsqueeze(0)

        encoded = self.transformer_encoder(src)

        pooled = torch.cat([
            encoded[:, 0, :], 
            encoded.mean(dim=1) 
        ], dim=-1)

        return self.output_proj(pooled)

class siRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, seq_len):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, seq_len, embedding_dim, n_head, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_tokens):
        encoded_sequence = self.encoder(x_tokens)
        return encoded_sequence

class mRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, seq_len):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, seq_len, embedding_dim, n_head, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_tokens):
        encoded_sequence = self.encoder(x_tokens)
        # pooled_embedding = self.pool(encoded_sequence.permute(0, 2, 1)).squeeze(-1)
        return encoded_sequence

class HeteroSAGE(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, layer_sizes, out_channels, metadata, sirna_seq_len, mrna_seq_len, sirna_other_feature_dim, mrna_other_feature_dim, interaction_feature_dim, dropout_rate=0.5):
        super().__init__()
        self.siRNA_encoder = siRNA_Encoder(vocab_size, embedding_dim, n_head, n_layers, seq_len=sirna_seq_len)
        self.mRNA_encoder = mRNA_Encoder(vocab_size, embedding_dim, n_head, n_layers, seq_len=mrna_seq_len)
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Calculate input dimensions for the first HeteroConv layer
        # Sequence embeddings are embedding_dim
        # Other features are sirna_other_feature_dim and mrna_other_feature_dim
        # Initial input dimensions after sequence embedding concatenation
        sirna_input_dim = sirna_other_feature_dim + embedding_dim
        mrna_input_dim  = mrna_other_feature_dim + embedding_dim
        interaction_input_dim = interaction_feature_dim  # unchanged

        for i in range(len(layer_sizes)):
            if i == 0:
                # First layer uses raw input sizes
                in_channels_dict = {
                    'siRNA': sirna_input_dim,
                    'mRNA': mrna_input_dim,
                    'interaction': interaction_input_dim,
                }
            else:
                in_channels_dict = {node_type: layer_sizes[i - 1] for node_type in self.node_types}

            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): GraphConv((in_channels_dict['siRNA'], in_channels_dict['interaction']), out_channels_i),
                ('mRNA', 'interacts_with', 'interaction'): GraphConv((in_channels_dict['mRNA'], in_channels_dict['interaction']), out_channels_i),
                ('interaction', 'rev_interacts_with', 'siRNA'): GraphConv((in_channels_dict['interaction'], in_channels_dict['siRNA']), out_channels_i),
                ('interaction', 'rev_interacts_with', 'mRNA'): GraphConv((in_channels_dict['interaction'], in_channels_dict['mRNA']), out_channels_i),
            }, aggr='mean')
            self.convs.append(conv)

            self.bns.append(nn.ModuleDict({
                node_type: BatchNorm1d(out_channels_i) for node_type in self.node_types
            }))
            self.dropouts.append(Dropout(dropout_rate))

        self.lin = Linear(layer_sizes[-1], out_channels)

    # def forward(self, x_dict, edge_index_dict):
    #     # Extract tokenized siRNA and mRNA input sequences
    #     sirna_tokens = x_dict['siRNA'].long()
    #     mrna_tokens = x_dict['mRNA'].long()

    #     # Pass through respective sequence encoders
    #     sirna_emb, _ = self.siRNA_encoder(sirna_tokens)
    #     mrna_emb, _ = self.mRNA_encoder(mrna_tokens)

    #     # # Concatenate sequence embeddings with other features
    #     # sirna_node_features = torch.cat([sirna_emb, sirna_other_features], dim=-1)
    #     # mrna_node_features = torch.cat([mrna_emb, mrna_other_features], dim=-1)

    #     # Prepare processed node feature dictionary for GNN
    #     x_dict_processed = {
    #         'siRNA': sirna_node_features,
    #         'mRNA': mrna_node_features,
    #         'interaction': x_dict['interaction']  # Interaction node features remain unchanged
    #     }
        
    def forward(self, x_dict, edge_index_dict, token_dict):
            # Get token sequences
        sirna_tokens = token_dict['siRNA']       # shape: [num_sirna_nodes, max_len]
        mrna_tokens  = token_dict['mRNA']        # shape: [num_mrna_nodes, max_len]

        # Encode
        sirna_emb = self.siRNA_encoder(sirna_tokens)
        mrna_emb  = self.mRNA_encoder(mrna_tokens)

        # Concatenate embeddings with existing features
        x_dict['siRNA'] = torch.cat([x_dict['siRNA'], sirna_emb], dim=-1)
        x_dict['mRNA']  = torch.cat([x_dict['mRNA'],  mrna_emb],  dim=-1)

        # GNN message passing
        x_dict_processed = x_dict
        # Run through GNN layers
        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict_processed = conv(x_dict_processed, edge_index_dict)
            for node_type in x_dict_processed:
                if node_type in bn_dict:
                    if x_dict_processed[node_type].size(0) > 1 or not self.training:
                        x_dict_processed[node_type] = bn_dict[node_type](x_dict_processed[node_type])
                x_dict_processed[node_type] = F.leaky_relu(x_dict_processed[node_type])
                x_dict_processed[node_type] = dropout(x_dict_processed[node_type])

        # Final MLP or prediction head over interaction nodes
        return self.lin(x_dict_processed['interaction'])


    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())

print(f"Loaded Parameters: {params}")
print(f"Max siRNA Length for Padding: {MAX_SIRNA_LENGTH}")
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
    test_file = "Simone_all_data.csv" # <--- IMPORTANT: Adjust this path to your actual test data file
    try:
        data = pd.read_csv(test_file)
        print(f"Successfully loaded test data from: {test_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {test_file}. Skipping fold {n}.")
        continue
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}. Skipping fold {n}.")
        continue

    # Substitute U to T for siRNA_seq to match mRNA_seq (DNA-like)

    data['siRNA_seq'] = data['siRNA_seq'].str.replace('U', 'T')

    # Check vocabulary size
    print("\n--- Checking Vocabulary for siRNA and mRNA Sequences ---")
    sirna_chars = set(''.join(data['siRNA_seq'].unique()))
    #mrna_chars = set(''.join(data['mRNA_seq_RNA-FM'].unique()))
    l = ['U', 'G', 'N', 'R', 'C', 'A']
    mrna_chars = set(l)
    all_chars = sirna_chars.union(mrna_chars)
    print(f"Unique characters in siRNA sequences: {sirna_chars}")
    print(f"Unique characters in mRNA sequences: {mrna_chars}")
    print(f"Combined unique characters: {all_chars}")
    vocab = ['<pad>'] + sorted(list(all_chars))
    vocab_size = len(vocab)
    print(f"Vocabulary: {vocab}")
    print(f"Vocab Size: {vocab_size}")

    print("\n--- Feature Processing ---")

    # 1. Tokenize siRNA sequences
    unique_sirna = data[['siRNA', 'siRNA_seq']].drop_duplicates(subset='siRNA')
    sirna_tokens_list = [tokenize_sequence(seq, MAX_SIRNA_LENGTH) for seq in unique_sirna['siRNA_seq']]
    sirna_tokens_df = pd.DataFrame(sirna_tokens_list, index=unique_sirna['siRNA'])

    # 2. Tokenize mRNA sequences
    unique_mrna = data[['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset='mRNA')
    mrna_tokens_list = [tokenize_sequence(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']]
    mrna_tokens_df = pd.DataFrame(mrna_tokens_list, index=unique_mrna['mRNA'])

    # 3. Positional encoding for interactions
    sirna_pos_encoding = []
    for idx, row in data.iterrows():
        mrna_start_pos = max(0, int(row['pos']))
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos,
            len(row['siRNA_seq']), # Use actual siRNA length for positional encoding context
            MAX_SIRNA_LENGTH,
            params["dmodel"] # This should match embedding_dim if used as part of it
        ))
    sirna_pos_encoding_df = pd.DataFrame(sirna_pos_encoding, index=data['siRNA'] + '_' + data['mRNA'])


    # 4. Thermodynamics
    sirna_thermo_feat_list = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                              for seq in data['siRNA_seq']]
    sirna_thermo_feat_df = pd.DataFrame(sirna_thermo_feat_list).reset_index(drop=True)
    temp_interaction_index = data['siRNA'] + '_' + data['mRNA']
    sirna_thermo_feat_df['index'] = temp_interaction_index
    sirna_thermo_feat_df = sirna_thermo_feat_df.set_index('index')

    # 5. Co-fold features
    con_feat = pd.read_csv("Simone_split_preprocess/con_matrix.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat_df.index).fillna(0)

    # 6. Self-fold features (siRNA)
    sirna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_tokens_df.index).fillna(0)

    # 7. Self-fold features (mRNA)
    mrna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_mRNA_matrix.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_tokens_df.index).fillna(0)

     # 8. GC percentage (variable length robust)
    sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data['siRNA_seq']], index=list(data['siRNA']))
    mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in unique_mrna['mRNA_seq_RNA-FM']], index=list(unique_mrna['mRNA']))

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

    # mrna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
    # mrna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
    # mrna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
    # mrna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
    # mrna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
    # mrna_k_mers = pd.concat([mrna_1_mer, mrna_2_mers, mrna_3_mers, mrna_4_mers, mrna_5_mers], axis=1)
    # mrna_k_mers.index = unique_mrna['mRNA']

    # 10. siRNA rules codes (PADDED to 19*3)
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))

    print("\n--- Assembling GNN Node Features ---")

    # Create heterogeneous graph data structure
    data_hetero = HeteroData()

    # Add node features:
    # siRNA and mRNA nodes will ONLY store tokenized sequences for the encoders.
    # Other features (self-fold) will be stored separately as global tensors,
    # and looked up during batch processing.
    # Concatenate all siRNA features
    sirna_all_feats = pd.concat([
        sirna_sfold_feat,
        sirna_GC,
        sirna_k_mers,
        sirna_pos_scores
    ], axis=1)
    data_hetero['siRNA'].x = torch.tensor(sirna_all_feats.values, dtype=torch.float)

    # Concatenate all mRNA features
    mrna_all_feats = pd.concat([
        mrna_sfold_feat,
        # mrna_k_mers
        mrna_GC
    ], axis=1)
    data_hetero['mRNA'].x = torch.tensor(mrna_all_feats.values, dtype=torch.float)

    data_hetero['interaction'].x = torch.tensor(
        pd.concat([sirna_thermo_feat_df, con_feat, sirna_pos_encoding_df], axis=1).values,
        dtype=torch.float
    )

    # Ensure index order of tokens matches the node features
    sirna_tokens_aligned = sirna_tokens_df.loc[sirna_all_feats.index].values
    mrna_tokens_aligned = mrna_tokens_df.loc[mrna_all_feats.index].values

    data_hetero['siRNA'].tokens = torch.tensor(sirna_tokens_aligned, dtype=torch.long)
    data_hetero['mRNA'].tokens = torch.tensor(mrna_tokens_aligned, dtype=torch.long)


    # Create edge indices
    sirna_name_to_idx = {name: i for i, name in enumerate(sirna_tokens_df.index)}
    mrna_name_to_idx = {name: i for i, name in enumerate(mrna_tokens_df.index)}
    interaction_name_to_idx = {name: i for i, name in enumerate(sirna_thermo_feat_df.index)} # Use thermo feat df index

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
            print(f"Warning: Skipping edge for {interaction_name} due to missing node in feature dataframes.")

    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

    print(f"\n--- PyG HeteroData Edge Index Shapes ---")
    for edge_type in data_hetero.edge_types:
        print(f"data_hetero{edge_type}.edge_index shape: {data_hetero[edge_type].edge_index.shape}")

    # Create train/dev masks and labels
    #train_idx_list = []
    dev_idx_list = []

    # for _, row in data_train.iterrows():
    #     interaction_name = f"{row['siRNA']}_{row['mRNA']}"
    #     if interaction_name in interaction_name_to_idx:
    #         train_idx_list.append(interaction_name_to_idx[interaction_name])

    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            dev_idx_list.append(interaction_name_to_idx[interaction_name])

    #train_idx = torch.tensor(train_idx_list, dtype=torch.long)
    dev_idx = torch.tensor(dev_idx_list, dtype=torch.long)

    # Create labels tensor
    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_name_to_idx:
            labels[interaction_name_to_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels

    print(f"\n--- PyG HeteroData Label Shape ---")
    print(f"data_hetero['interaction'].y shape: {data_hetero['interaction'].y.shape}")

    # Initialize NeighborLoaders
    # Note: NeighborLoader automatically creates `n_id` for each node type in the batch.
    # We will use these `n_id` to lookup the corresponding self-fold features from the global tensors.
    # train_loader = NeighborLoader(
    #     data_hetero,
    #     num_neighbors=params["hop_samples"],
    #     batch_size=params["batch_size"],
    #     input_nodes=('interaction', train_idx),
    #     shuffle=True,
    #     subgraph_type='induced',
    #     filter_per_worker=params["filter_per_worker"],
    #     num_workers=params["num_workers"]
    # )

    val_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', dev_idx),
        shuffle=False,
        subgraph_type='induced',
        filter_per_worker=params["filter_per_worker"],
        num_workers=params["num_workers"]
    )

    # Determine feature dimensions for the HeteroSAGE model
    sirna_other_feature_dim = sirna_all_feats.shape[1]
    mrna_other_feature_dim = mrna_all_feats.shape[1]
    interaction_feature_dim = data_hetero['interaction'].x.shape[1]

    # Model Initialization
    # device is already defined globally now
    model = HeteroSAGE(
        vocab_size=vocab_size,
        embedding_dim=params["embedding_dim"],
        n_head=params["n_head"],
        n_layers=params["n_layers"],
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        sirna_seq_len=MAX_SIRNA_LENGTH,
        mrna_seq_len=MAX_MRNA_LENGTH,
        sirna_other_feature_dim=sirna_other_feature_dim, # Pass new dims
        mrna_other_feature_dim=mrna_other_feature_dim,   # Pass new dims
        interaction_feature_dim=interaction_feature_dim,
        dropout_rate=params["dropout"]
    ).to(device)

    data_hetero = data_hetero.to(device) # Move the entire graph to the device

    # Load the saved weights
    model_path = f"best256_model_fold{n}_enc.pt" # <--- IMPORTANT: Ensure this path is correct for your saved models
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

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params["scheduler_factor"], patience=params["scheduler_patience"], min_lr=params["min_lr"], verbose=True)
    # criterion = nn.HuberLoss(delta=params["huber_delta"]) if params["loss"] == "huber" else nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss() if params["loss"] == "mse" else nn.L1Loss()

    # def train():
    #     model.train()
    #     total_loss = 0
    #     scaler = GradScaler()  # For mixed precision
    #     for batch in train_loader:
    #         batch = batch.to(device)
    #         optimizer.zero_grad()

    #         # Lookup self-fold features for the current batch's nodes
    #         # These tensors are now correctly moved to device globally
    #         batch_siRNA_sfold = sirna_sfold_tensor[batch['siRNA'].n_id]
    #         batch_mRNA_sfold = mrna_sfold_tensor[batch['mRNA'].n_id]

    #         with autocast():  # Enable mixed precision
    #             out = model(batch.x_dict, batch.edge_index_dict, batch_siRNA_sfold, batch_mRNA_sfold)
    #             loss = criterion(out.squeeze(-1), batch['interaction'].y)
    #             l1_lambda = params.get("l1_lambda", 0.0)
    #             if l1_lambda > 0:
    #                 loss += l1_lambda * model.l1_loss()
    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["clip_grad_norm"])
    #         scaler.step(optimizer)
    #         scaler.update()
    #         total_loss += loss.item()
    #     return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            batch = batch.to(device)
            token_dict = {
                    'siRNA': batch['siRNA'].tokens,   # [num_siRNA_nodes, max_len]
                    'mRNA': batch['mRNA'].tokens      # [num_mRNA_nodes, max_len]
                }

            # with autocast():  # Enable mixed precision
            out = model(batch.x_dict, batch.edge_index_dict, token_dict=token_dict)
            preds.append(out.cpu())
            truths.append(batch['interaction'].y.cpu())
        return torch.cat(preds), torch.cat(truths)

    test_pred, test_true = validate(val_loader)
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
    #if len(np.unique(test_true > 0.7)) > 1: # Check if there are at least two unique binary classes
    auc = roc_auc_score((test_true>=0.7).astype(int), test_pred)
    print(f"AUC: {auc:.4f}")
    score_auc.append(auc)
    # else:
    #     print(f"AUC: Not enough unique classes for AUC calculation (all 'efficacy' values are on one side of 0.7 threshold).")

    print(f"Fold {n} finished!")