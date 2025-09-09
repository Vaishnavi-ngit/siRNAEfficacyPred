import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" # Use a single GPU as per Claude's suggestion, or "1,2" if multi-GPU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
# IMPORTANT: Use PyG's GATv2Conv
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score # <-- ADDED: roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import scipy.stats
import json
import math
from torch.nn import Dropout, LayerNorm # LayerNorm is now from nn, not custom
from torch.cuda.amp import GradScaler, autocast
import Utils as utils1 # Your original utils file

# --- Load parameters ---
# Use your original param file for now to avoid confusion
try:
    with open("param.json", 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print("siRNA_param_pytorch.json not found, using default parameters.")
    params = {}

# Define MAX_SIRNA_LENGTH and MAX_MRNA_LENGTH from parameters
MAX_SIRNA_LENGTH = params.setdefault("sirna_length", 19)
MAX_MRNA_LENGTH = params.setdefault("max_mrna_len", 128) # You had 128 in utils1, make consistent
params.setdefault("embedding_dim", 512) # Larger embedding as per Claude's suggestion
params.setdefault("dmodel", params["embedding_dim"]) # Ensure dmodel matches embedding_dim
params.setdefault("n_head", 8)
params.setdefault("n_layers", 3)
params.setdefault("gat_heads", 8) # For GATv2Conv
params.setdefault("hinsage_layer_sizes", [512, 256, 128])
params.setdefault("dropout", 0.1)
params.setdefault("batch_size", 64)
params.setdefault("hop_samples", [20, 15])
params.setdefault("clip_grad_norm", 0.5)
params.setdefault("lr", 0.001)
params.setdefault("weight_decay", 0.001)
params.setdefault("epochs", 300)
params.setdefault("early_stopping_patience", 30)
params.setdefault("loss", "mse")
params.setdefault("l1_lambda", 1e-6) # For L1 regularization

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Tokenization Function (from OG code, with U to T handling) ---
def tokenize_sequence(seq, max_len):
    # Ensure U is handled consistently with T
    seq = seq.replace('U', 'T').upper() # Convert U to T, and uppercase
    token_map = {
        'A': 1, 'T': 2, 'G': 3, 'C': 4,
    } # 0 reserved for <pad>, other chars will also be 0
    tokens = [token_map.get(base, 0) for base in seq]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

# --- MODIFIED: EnhancedTransformerEncoder to accept additional features ---
class EnhancedTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, max_len, dropout, additional_feature_dim=0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
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

        # Output projection takes (CLS + Mean) embedding and additional features
        # CLS + Mean pooling gives 2 * d_model
        # Additional features are additional_feature_dim
        # So total input to output_proj is 2 * d_model + additional_feature_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2 + additional_feature_dim, d_model), # Reduced first layer of output_proj
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model) # Output is just d_model, as per the original HeteroSAGE input
        )

    def forward(self, x_tokens, additional_features=None):
        seq_len = x_tokens.size(1)
        src = self.embedding(x_tokens) * math.sqrt(self.d_model)
        src = src + self.pos_encoding[:seq_len].unsqueeze(0)

        encoded = self.transformer_encoder(src)

        # Global average pooling + CLS token
        pooled = torch.cat([
            encoded[:, 0, :], # CLS token (assuming first token is CLS or global rep)
            encoded.mean(dim=1) # Average pooling
        ], dim=-1)

        # Concatenate with additional features if provided
        if additional_features is not None:
            pooled = torch.cat([pooled, additional_features], dim=-1)

        return self.output_proj(pooled)

# --- FIXED: EnhancedHeteroGNN with proper residual projection handling ---
class EnhancedHeteroGNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, gat_heads, layer_sizes,
                 out_channels, metadata, sirna_seq_len, mrna_seq_len, # Added mrna_seq_len
                 sirna_other_feature_dim, mrna_other_feature_dim, # These are now ONLY self-fold, rules etc.
                 sirna_seq_feat_dim, mrna_seq_feat_dim, # New: GC + k-mers
                 interaction_feature_dim, dropout_rate):
        super().__init__()

        # Enhanced sequence encoder for siRNA (takes GC + k-mers)
        self.siRNA_encoder = EnhancedTransformerEncoder(
            vocab_size, embedding_dim, n_head, n_layers, sirna_seq_len, dropout_rate,
            additional_feature_dim=sirna_seq_feat_dim
        )
        # Enhanced sequence encoder for mRNA (takes GC + k-mers, if applicable for mRNA)
        self.mRNA_encoder = EnhancedTransformerEncoder( # Added mRNA encoder
            vocab_size, embedding_dim, n_head, n_layers, mrna_seq_len, dropout_rate,
            additional_feature_dim=mrna_seq_feat_dim
        )

        # Feature normalization layers (for the 'other' features, not sequence embeddings yet)
        self.feature_norms = nn.ModuleDict({
            'siRNA': LayerNorm(sirna_other_feature_dim),
            'mRNA': LayerNorm(mrna_other_feature_dim),
            'interaction': LayerNorm(interaction_feature_dim)
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # FIXED: Pre-compute all residual projections during initialization
        self.residual_projs = nn.ModuleDict()

        # Input dimensions for GNN layers:
        # Initial siRNA input: sirna_other_feature_dim + (d_model from its encoder)
        # Initial mRNA input: mrna_other_feature_dim + (d_model from its encoder)
        # Initial interaction input: interaction_feature_dim
        sirna_initial_gnn_dim = sirna_other_feature_dim + embedding_dim
        mrna_initial_gnn_dim = mrna_other_feature_dim + embedding_dim

        # Track input dimensions for each layer and node type
        layer_input_dims = {}
        
        for i, size in enumerate(layer_sizes):
            if i == 0:
                in_channels = {
                    'siRNA': sirna_initial_gnn_dim,
                    'mRNA': mrna_initial_gnn_dim,
                    'interaction': interaction_feature_dim,
                }
            else:
                in_channels = {key: layer_sizes[i-1] for key in metadata[0]}
            
            # Store input dimensions for this layer
            layer_input_dims[i] = in_channels.copy()

            # Use PyG's GATv2Conv with residual and LayerNorm handled internally (or externally if needed)
            # PyG's GATv2Conv handles multi-head concatenation and dropout
            conv = HeteroConv({
                edge_type: GATv2Conv(
                    in_channels=(in_channels[edge_type[0]], in_channels[edge_type[2]]), # GATv2 takes tuple for heterogeneous input
                    out_channels=size // gat_heads, # Output per head
                    heads=gat_heads,
                    add_self_loops=False, # Assuming no self-loops on interaction nodes needed explicitly
                    dropout=dropout_rate,
                    bias=False # Bias is handled by layer norm in pre-norm GAT
                )
                for edge_type in metadata[1]
            }, aggr='mean')

            self.convs.append(conv)
            
            # LayerNorms AFTER the conv output (post-norm) in this setup
            self.norms.append(nn.ModuleDict({
                node_type: LayerNorm(size) for node_type in metadata[0]
            }))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # FIXED: Pre-compute residual projections for this layer
            for node_type in metadata[0]:
                # The input to the current GNN layer (for residual) is the output of the *previous* layer.
                # For the first layer (i=0), the residual comes from the initial concatenated features.
                # For subsequent layers, it comes from the output of the (i-1)-th GNN layer.
                if i == 0:
                    residual_input_dim = in_channels[node_type] # Use the initial combined dimension
                else:
                    residual_input_dim = layer_sizes[i-1] # Use the output dimension of the previous layer

                output_dim = size # The output dimension of the current GNN layer

                # Only create projection if dimensions don't match
                if residual_input_dim != output_dim:
                    proj_key = f"{node_type}_layer_{i}"
                    self.residual_projs[proj_key] = nn.Linear(residual_input_dim, output_dim, bias=False)


        # Enhanced prediction head
        final_dim = layer_sizes[-1]
        self.prediction_head = nn.Sequential(
            nn.Linear(final_dim, final_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_dim * 2, final_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_dim // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, token_dict, seq_feat_dict):
        # 1. Normalize initial node features (non-sequence based)
        for node_type, x in x_dict.items():
            if node_type in self.feature_norms:
                x_dict[node_type] = self.feature_norms[node_type](x)

        # 2. Enhanced sequence encoding with additional features
        # siRNA: Pass tokens and its GC/k-mer features
        siRNA_emb = self.siRNA_encoder(token_dict['siRNA'], seq_feat_dict['siRNA'])
        x_dict['siRNA'] = torch.cat([x_dict['siRNA'], siRNA_emb], dim=-1)

        # mRNA: Pass tokens and its GC/k-mer features
        mRNA_emb = self.mRNA_encoder(token_dict['mRNA'], seq_feat_dict['mRNA'])
        x_dict['mRNA'] = torch.cat([x_dict['mRNA'], mRNA_emb], dim=-1)

        # 3. Graph convolution with residual connections and post-normalization/activation
        for i, (conv, norm_dict, dropout) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            # Store the input to this layer for residual connections
            x_residual = {k: v.clone() for k, v in x_dict.items()}
            
            # Apply convolution
            x_dict = conv(x_dict, edge_index_dict)

            # Apply residual connections, normalization, and activation
            for node_type, x in x_dict.items():
                residual = x_residual[node_type]
                
                # Apply residual projection if needed
                proj_key = f"{node_type}_layer_{i}"
                if proj_key in self.residual_projs:
                    residual = self.residual_projs[proj_key](residual)
                
                # Residual connection
                x = x + residual
                
                # Apply normalization, activation, and dropout
                # Handle single-sample batches during validation (LayerNorm needs >1 sample for training stats)
                if x.size(0) > 1 or not self.training:
                    x = norm_dict[node_type](x)
                
                x = F.gelu(x)
                x = dropout(x)
                x_dict[node_type] = x

        # 4. Final prediction head
        return self.prediction_head(x_dict['interaction'])

# --- Revert to OG K-Fold Data Loading and Processing ---
def main():
    print("Starting Enhanced siRNA Efficacy Prediction Model (Integrated)")
    print(f"Using device: {DEVICE}")
    print(f"Parameters: {params}")

    NUM_FOLDS = 10 # Your original number of folds
    all_fold_results = []

    for n in range(NUM_FOLDS):
        print(f"\n--- Processing fold {n+1}/{NUM_FOLDS} ---")

        split_dir = f"siRNA_split_datasets/split{n}/"
        if not os.path.exists(split_dir):
            print(f"Error: Directory '{split_dir}' not found. Please ensure your data splits are correctly organized.")
            print("Skipping fold processing.")
            continue

        try:
            data_train_raw = pd.read_csv(os.path.join(split_dir, "train.csv"))
            data_dev_raw = pd.read_csv(os.path.join(split_dir, "dev.csv"))
        except FileNotFoundError as e:
            print(f"Error: Data file not found in '{split_dir}'. {e}")
            print("Skipping fold processing.")
            continue

        # Convert U to T for consistency before feature generation
        for df in [data_train_raw, data_dev_raw]:
            df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T').str.upper() # Ensure uppercase too
            df['mRNA_seq_RNA-FM'] = df['mRNA_seq_RNA-FM'].str.upper()

        data_combined = pd.concat([data_train_raw, data_dev_raw], axis=0).reset_index(drop=True)

        print("\n--- Feature Processing (from OG code) ---")

        # Create unique mappings for IDs
        unique_sirna_ids = data_combined['siRNA'].drop_duplicates().reset_index(drop=True)
        unique_mrna_ids = data_combined['mRNA'].drop_duplicates().reset_index(drop=True)
        # Create a combined unique sequence mapping for features that depend on sequence content
        unique_sirna_seq_map = data_combined[['siRNA', 'siRNA_seq']].drop_duplicates(subset='siRNA').set_index('siRNA')
        unique_mrna_seq_map = data_combined[['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset='mRNA').set_index('mRNA')

        # --- Features for EnhancedTransformerEncoder (GC + K-mers) ---
        # These features are calculated per unique sequence and will be passed to the encoder
        sirna_gc_kmers = []
        for seq_id in unique_sirna_seq_map.index:
            seq = unique_sirna_seq_map.loc[seq_id, 'siRNA_seq']
            gc = utils1.countGC(seq)
            kmer_freqs = utils1.get_kmer_freq(seq, k=1) # 1-mer
            kmer_freqs.extend(utils1.get_kmer_freq(seq, k=2)) # 2-mer
            # Add more k-mers if desired
            sirna_gc_kmers.append([gc] + kmer_freqs)
        sirna_gc_kmers_df = pd.DataFrame(sirna_gc_kmers, index=unique_sirna_seq_map.index)

        mrna_gc_kmers = []
        for seq_id in unique_mrna_seq_map.index:
            seq = unique_mrna_seq_map.loc[seq_id, 'mRNA_seq_RNA-FM']
            gc = utils1.countGC(seq)
            kmer_freqs = utils1.get_kmer_freq(seq, k=1) # 1-mer
            kmer_freqs.extend(utils1.get_kmer_freq(seq, k=2)) # 2-mer
            # Add more k-mers if desired
            mrna_gc_kmers.append([gc] + kmer_freqs)
        mrna_gc_kmers_df = pd.DataFrame(mrna_gc_kmers, index=unique_mrna_seq_map.index)


        # --- Other Node Features (siRNA.x, mRNA.x, interaction.x) ---
        # 1. siRNA tokenized sequences (for Transformer)
        sirna_tokens_aligned_list = [tokenize_sequence(seq, MAX_SIRNA_LENGTH)
                                     for seq in unique_sirna_seq_map['siRNA_seq']]
        sirna_tokens_tensor = torch.tensor(sirna_tokens_aligned_list, dtype=torch.long)
        # 2. mRNA tokenized sequences (for Transformer)
        mrna_tokens_aligned_list = [tokenize_sequence(seq, MAX_MRNA_LENGTH) # Use actual mRNA_seq_RNA-FM here
                                     for seq in unique_mrna_seq_map['mRNA_seq_RNA-FM']]
        mrna_tokens_tensor = torch.tensor(mrna_tokens_aligned_list, dtype=torch.long)

        # 3. siRNA other features (self-fold, rules scores)
        sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
        sirna_sfold_feat = sirna_sfold_feat.reindex(unique_sirna_seq_map.index).fillna(0)
        sirna_pos_scores = pd.DataFrame([utils1.rules_scores(seq, params["sirna_length"]) for seq in unique_sirna_seq_map['siRNA_seq']],
                                         index=unique_sirna_seq_map.index)
        sirna_other_feats = pd.concat([sirna_sfold_feat, sirna_pos_scores], axis=1).fillna(0) # Exclude GC/k-mers here

        # 4. mRNA other features (self-fold, MP-RNA embedding)
        mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
        mrna_sfold_feat = mrna_sfold_feat.reindex(unique_mrna_seq_map.index).fillna(0)
        # Check and use MP-RNA embeddings if available in utils1
        mrna_rnafm_embeddings = pd.DataFrame([utils1.get_mp_rna_sequence_embedding(seq) for seq in unique_mrna_seq_map['mRNA_seq_RNA-FM']],
                                             index=unique_mrna_seq_map.index)
        mrna_other_feats = pd.concat([mrna_sfold_feat, mrna_rnafm_embeddings], axis=1).fillna(0)


        # 5. Interaction features (thermo, co-fold, positional encoding)
        # Index for interaction features will be a composite of siRNA_ID_mRNA_ID
        interaction_index_map = data_combined.apply(lambda row: f"{row['siRNA']}_{row['mRNA']}", axis=1)

        sirna_thermo_feat_list = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                                  for seq in data_combined['siRNA_seq']]
        sirna_thermo_feat_df = pd.DataFrame(sirna_thermo_feat_list).set_index(interaction_index_map)

        con_feat = pd.read_csv("siRNA_split_preprocess/full_con_matrix.txt", header=None, index_col=0)
        con_feat = con_feat.reindex(interaction_index_map).fillna(0)

        sirna_pos_encoding = []
        for idx, row in data_combined.iterrows():
            mrna_start_pos = max(0, int(row['pos']))
            sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
                mrna_start_pos,
                len(row['siRNA_seq']),
                MAX_SIRNA_LENGTH,
                params["dmodel"]
            ))
        sirna_pos_encoding_df = pd.DataFrame(sirna_pos_encoding).set_index(interaction_index_map)

        interaction_all_feats = pd.concat([sirna_thermo_feat_df, con_feat, sirna_pos_encoding_df], axis=1).fillna(0)
        # NOTE: Claude's 'binding_feats' called get_binding_site_features, which is a placeholder.
        # It's omitted here to avoid adding empty features unless implemented.


        # Feature Scaling (RobustScaler recommended by Claude)
        print("\nScaling features...")
        scaler_siRNA = RobustScaler()
        scaler_mRNA = RobustScaler()
        scaler_interaction = RobustScaler()
        scaler_siRNA_gc_kmer = RobustScaler()
        scaler_mRNA_gc_kmer = RobustScaler()

        sirna_other_feats_scaled = pd.DataFrame(
            scaler_siRNA.fit_transform(sirna_other_feats),
            index=sirna_other_feats.index,
            columns=sirna_other_feats.columns
        )
        mrna_other_feats_scaled = pd.DataFrame(
            scaler_mRNA.fit_transform(mrna_other_feats),
            index=mrna_other_feats.index,
            columns=mrna_other_feats.columns
        )
        interaction_all_feats_scaled = pd.DataFrame(
            scaler_interaction.fit_transform(interaction_all_feats),
            index=interaction_all_feats.index,
            columns=interaction_all_feats.columns
        )
        sirna_gc_kmers_scaled = pd.DataFrame(
            scaler_siRNA_gc_kmer.fit_transform(sirna_gc_kmers_df),
            index=sirna_gc_kmers_df.index,
            columns=sirna_gc_kmers_df.columns
        )
        mrna_gc_kmers_scaled = pd.DataFrame(
            scaler_mRNA_gc_kmer.fit_transform(mrna_gc_kmers_df),
            index=mrna_gc_kmers_df.index,
            columns=mrna_gc_kmers_df.columns
        )


        # --- Construct HeteroData for this fold ---
        data_hetero = HeteroData()

        # Node features (excluding sequence features which go to encoders)
        # Ensure order of unique_siRNA_ids/unique_mRNA_ids matches the index of features
        data_hetero['siRNA'].x = torch.tensor(sirna_other_feats_scaled.reindex(unique_sirna_ids).values, dtype=torch.float)
        data_hetero['mRNA'].x = torch.tensor(mrna_other_feats_scaled.reindex(unique_mrna_ids).values, dtype=torch.float)
        data_hetero['interaction'].x = torch.tensor(interaction_all_feats_scaled.values, dtype=torch.float)

        # Tokenized sequences (for Transformer encoders)
        data_hetero['siRNA'].tokens = sirna_tokens_tensor.clone().detach() # Use the pre-computed tensor
        data_hetero['mRNA'].tokens = mrna_tokens_tensor.clone().detach()

        # GC + k-mer features (for Transformer encoders)
        # Ensure these are indexed correctly to align with unique_siRNA_ids / unique_mRNA_ids
        data_hetero['siRNA'].seq_features = torch.tensor(sirna_gc_kmers_scaled.reindex(unique_sirna_ids).values, dtype=torch.float)
        data_hetero['mRNA'].seq_features = torch.tensor(mrna_gc_kmers_scaled.reindex(unique_mrna_ids).values, dtype=torch.float)


        # Create edge indices (similar to OG code)
        siRNA_name_to_idx = {name: i for i, name in enumerate(unique_sirna_ids)}
        mRNA_name_to_idx = {name: i for i, name in enumerate(unique_mrna_ids)}
        interaction_name_to_idx = {name: i for i, name in enumerate(interaction_all_feats_scaled.index)} # Corrected to use scaled interaction features index

        edge_index_siRNA_to_interaction = []
        edge_index_mRNA_to_interaction = []
        interaction_labels_in_order = [] # Collect labels in the order they appear in interaction_all_feats_scaled

        for _, row in data_combined.iterrows():
            siRNA_name = row['siRNA']
            mRNA_name = row['mRNA']
            interaction_name = f"{siRNA_name}_{mRNA_name}"

            if (siRNA_name in siRNA_name_to_idx and
                mRNA_name in mRNA_name_to_idx and
                interaction_name in interaction_name_to_idx): # Ensure interaction exists in features
                edge_index_siRNA_to_interaction.append([siRNA_name_to_idx[siRNA_name], interaction_name_to_idx[interaction_name]])
                edge_index_mRNA_to_interaction.append([mRNA_name_to_idx[mRNA_name], interaction_name_to_idx[interaction_name]])
            else:
                # This warning might indicate issues with indexing or missing data
                print(f"Warning: Skipping edge for {interaction_name} due to missing node in feature dataframes or index mismatch.")

        data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
            edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
        data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
            edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
        # Add reverse edges for HeteroConv
        data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
        data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

        # Labels (aligned with interaction.x order)
        # Create a mapping from (siRNA, mRNA) tuple to efficacy
        efficacy_map = data_combined.set_index(['siRNA', 'mRNA'])['efficacy'].to_dict()
        
        ordered_labels = []
        # Iterate over the index of interaction_all_feats_scaled to ensure correct order
        for interaction_key in interaction_all_feats_scaled.index:
            try:
                # Split the compound key back to (siRNA_ID, mRNA_ID) to query efficacy_map
                siRNA_id, mRNA_id = interaction_key.split('_', 1) # Split only on the first '_'
                ordered_labels.append(efficacy_map[(siRNA_id, mRNA_id)])
            except KeyError:
                print(f"CRITICAL WARNING: Efficacy for interaction {interaction_key} not found in map. This may cause label/feature mismatch!")
                pass 
        # Added assertion to catch label/feature dimension mismatch early
        assert len(ordered_labels) == interaction_all_feats_scaled.shape[0], \
            f"Label count mismatch! Expected {interaction_all_feats_scaled.shape[0]}, got {len(ordered_labels)}."

        data_hetero['interaction'].y = torch.tensor(ordered_labels, dtype=torch.float)


        train_interaction_indices = [interaction_name_to_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train_raw.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_name_to_idx]
        dev_interaction_indices = [interaction_name_to_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev_raw.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_name_to_idx]

        train_idx = torch.tensor(train_interaction_indices, dtype=torch.long)
        dev_idx = torch.tensor(dev_interaction_indices, dtype=torch.long)

        print(f"HeteroData Graph created: {data_hetero}")
        print(f"Number of siRNA nodes: {data_hetero['siRNA'].num_nodes}")
        print(f"Number of mRNA nodes: {data_hetero['mRNA'].num_nodes}")
        print(f"Number of interaction nodes: {data_hetero['interaction'].num_nodes}")

        train_loader = NeighborLoader(
            data_hetero,
            num_neighbors=params["hop_samples"],
            batch_size=params["batch_size"],
            input_nodes=('interaction', train_idx),
            shuffle=True,
            subgraph_type='induced',
            num_workers=0
        )

        val_loader = NeighborLoader(
            data_hetero,
            num_neighbors=params["hop_samples"],
            batch_size=params["batch_size"],
            input_nodes=('interaction', dev_idx),
            shuffle=False,
            subgraph_type='induced',
            num_workers=0
        )

        sirna_other_feature_dim = data_hetero['siRNA'].x.shape[1]
        mrna_other_feature_dim = data_hetero['mRNA'].x.shape[1]
        interaction_feature_dim = data_hetero['interaction'].x.shape[1]
        sirna_seq_feat_dim = data_hetero['siRNA'].seq_features.shape[1]
        mrna_seq_feat_dim = data_hetero['mRNA'].seq_features.shape[1]

        model = EnhancedHeteroGNN(
            vocab_size=5,
            embedding_dim=params["embedding_dim"],
            n_head=params["n_head"],
            n_layers=params["n_layers"],
            gat_heads=params["gat_heads"],
            layer_sizes=params["hinsage_layer_sizes"],
            out_channels=1,
            metadata=data_hetero.metadata(),
            sirna_seq_len=MAX_SIRNA_LENGTH,
            mrna_seq_len=MAX_MRNA_LENGTH,
            sirna_other_feature_dim=sirna_other_feature_dim,
            mrna_other_feature_dim=mrna_other_feature_dim,
            sirna_seq_feat_dim=sirna_seq_feat_dim,
            mrna_seq_feat_dim=mrna_seq_feat_dim,
            interaction_feature_dim=interaction_feature_dim,
            dropout_rate=params["dropout"]
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
        criterion = nn.MSELoss() if params["loss"] == "mse" else nn.SmoothL1Loss()
        scaler = GradScaler()

        best_val_loss_fold = float('inf')
        patience_counter = 0

        for epoch in range(params["epochs"]):
            # Training
            train_loss = 0
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    token_dict = {
                        'siRNA': batch['siRNA'].tokens,
                        'mRNA': batch['mRNA'].tokens # Pass mRNA tokens too
                    }
                    seq_feat_dict = {
                        'siRNA': batch['siRNA'].seq_features,
                        'mRNA': batch['mRNA'].seq_features
                    }
                    out = model(batch.x_dict, batch.edge_index_dict, token_dict, seq_feat_dict).squeeze()
                    loss = criterion(out, batch['interaction'].y)

                    l1_lambda = params.get("l1_lambda", 0.0)
                    if l1_lambda > 0:
                        l1_reg = torch.tensor(0.0, device=DEVICE)
                        for param in model.parameters():
                            l1_reg += torch.norm(param, 1)
                        loss += l1_lambda * l1_reg

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            val_loss = 0
            val_preds_list, val_targets_list = [], [] # Use _list to avoid confusion with numpy arrays
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    token_dict = {
                        'siRNA': batch['siRNA'].tokens,
                        'mRNA': batch['mRNA'].tokens
                    }
                    seq_feat_dict = {
                        'siRNA': batch['siRNA'].seq_features,
                        'mRNA': batch['mRNA'].seq_features
                    }
                    out = model(batch.x_dict, batch.edge_index_dict, token_dict, seq_feat_dict).squeeze()
                    loss = criterion(out, batch['interaction'].y)
                    val_loss += loss.item()
                    val_preds_list.extend(out.cpu().numpy())
                    val_targets_list.extend(batch['interaction'].y.cpu().numpy())
                val_loss /= len(val_loader)

            val_preds = np.array(val_preds_list)
            val_targets = np.array(val_targets_list)

            # Calculate PCC (Pearson Correlation Coefficient)
            val_pcc = scipy.stats.pearsonr(val_targets, val_preds)[0]

            # Calculate AUC (requires binarization for regression)
            # We will binarize around the median of the true labels for the validation set
            # This makes AUC interpretable as "how well the model ranks predictions relative to the median"
            efficacy_threshold = np.median(val_targets)
            bin_val_targets = (val_targets >= efficacy_threshold).astype(int)
            
            # Ensure there are at least two classes in bin_val_targets for AUC calculation
            if len(np.unique(bin_val_targets)) > 1:
                val_auc = roc_auc_score(bin_val_targets, val_preds)
            else:
                val_auc = np.nan # Cannot calculate AUC if only one class present

            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            
            scheduler.step() # CosineAnnealingWarmRestarts steps every epoch, not based on loss

            # Early stopping
            if val_loss < best_val_loss_fold:
                best_val_loss_fold = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold{n}.pth')
            else:
                patience_counter += 1
                if patience_counter >= params["early_stopping_patience"]:
                    print(f"Early stopping triggered for fold {n+1} after {epoch + 1} epochs.")
                    break

            if epoch % 10 == 0 or epoch == params["epochs"] - 1:
                print(f"Fold {n+1}, Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | "
                      f"Val PCC: {val_pcc:.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Load best model for this fold and evaluate
        model.load_state_dict(torch.load(f'best_model_fold{n}.pth'))
        final_val_loss = 0
        final_preds_list, final_targets_list = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                token_dict = {
                    'siRNA': batch['siRNA'].tokens,
                    'mRNA': batch['mRNA'].tokens
                }
                seq_feat_dict = {
                    'siRNA': batch['siRNA'].seq_features,
                    'mRNA': batch['mRNA'].seq_features
                }
                out = model(batch.x_dict, batch.edge_index_dict, token_dict, seq_feat_dict).squeeze()
                final_val_loss += criterion(out, batch['interaction'].y).item()
                final_preds_list.extend(out.cpu().numpy())
                final_targets_list.extend(batch['interaction'].y.cpu().numpy())
            final_val_loss /= len(val_loader)

        final_preds = np.array(final_preds_list)
        final_targets = np.array(final_targets_list)

        final_rmse = np.sqrt(mean_squared_error(final_targets, final_preds))
        final_pcc = scipy.stats.pearsonr(final_targets, final_preds)[0]

        efficacy_threshold_final = np.median(final_targets)
        bin_final_targets = (final_targets >= efficacy_threshold_final).astype(int)
        if len(np.unique(bin_final_targets)) > 1:
            final_auc = roc_auc_score(bin_final_targets, final_preds)
        else:
            final_auc = np.nan

        fold_results = {
            'fold': n,
            'val_loss': final_val_loss,
            'val_rmse': final_rmse,
            'val_correlation': final_pcc,
            'val_auc': final_auc # <-- ADDED: val_auc to results
        }
        all_fold_results.append(fold_results)
        print(f"Fold {n+1} Final Results: Val Loss: {final_val_loss:.4f}, Val RMSE: {final_rmse:.4f}, Val PCC: {final_pcc:.4f}, Val AUC: {final_auc:.4f}") # <-- ADDED: Val AUC to print

    print("\n--- All Folds Complete ---")
    avg_rmse = np.mean([r['val_rmse'] for r in all_fold_results])
    avg_pcc = np.mean([r['val_correlation'] for r in all_fold_results])
    # Filter out NaNs if any fold had issues calculating AUC
    valid_aucs = [r['val_auc'] for r in all_fold_results if not np.isnan(r['val_auc'])]
    avg_auc = np.mean(valid_aucs) if valid_aucs else np.nan # <-- ADDED: avg_auc calculation

    print(f"Average RMSE across {NUM_FOLDS} folds: {avg_rmse:.4f}")
    print(f"Average PCC across {NUM_FOLDS} folds: {avg_pcc:.4f}")
    print(f"Average AUC across {NUM_FOLDS} folds: {avg_auc:.4f}") # <-- ADDED: avg_auc print

    final_summary = {
        'average_rmse': avg_rmse,
        'average_correlation': avg_pcc,
        'average_auc': avg_auc, # <-- ADDED: average_auc to summary
        'all_fold_results': all_fold_results,
        'parameters': params
    }
    with open('training_summary_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    print("Training summary saved to training_summary_results.json")

if __name__ == "__main__":
    main()