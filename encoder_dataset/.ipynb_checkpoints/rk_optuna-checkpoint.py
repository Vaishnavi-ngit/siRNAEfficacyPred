import os
# Ensure CUDA_VISIBLE_DEVICES is set before any torch operations that might initialize CUDA
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
from sklearn.metrics import mean_squared_error
import scipy.stats
import json
import math
from torch.cuda.amp import GradScaler, autocast # Import for mixed precision training

# --- Optuna Imports ---
import optuna
from optuna.storages import RDBStorage # Required for persistent storage

torch.cuda.manual_seed_all(42)
import BugUtils as utils1

# --- Load parameters ---
with open("param.json", 'r') as f:
    params = json.load(f)

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

params.setdefault("embedding_dim", 256)  # Default embedding dimension
params.setdefault("n_head", 8)          # Default number of attention heads (will be tuned)
params.setdefault("n_layers", 4)        # Default number of transformer layers (NOT TUNED)       # Ensure dmodel is set for positional encoding in utils1 (NOT TUNED)


# Define device here, at the top level, so it's accessible globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def tokenize_sequence(seq, max_len):
    token_map = {
        'A': 1, 'T': 2, 'U':2, 'G': 3, 'C': 4,
        'U': 5, 'N': 6, 'R': 7, '-': 8  # Handle additional characters
    }  # 0 reserved for <pad>
    tokens = [token_map.get(base, 0) for base in seq]  # Unknown bases get 0 (<pad>)
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))  # Pad with 0
    else:
        tokens = tokens[:max_len]  # Truncate if longer
    return tokens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        # The input x here is usually the token embedding, so its shape is [batch_size, seq_len, d_model]
        # We need positional encoding for up to seq_len
        batch_size, seq_len, _ = x.size()
        return self.pe[:seq_len, :].to(x.device)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=0)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(tok_emb)
        return tok_emb + pos_emb

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        d_tensor = q.size(-1)
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02) # Original dropout for FFN, keep fixed as per your instruction

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x) # Original uses ReLU twice, keeping this
        x = self.dropout(x) # Original uses dropout twice, keeping this
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out, attention

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout = nn.Dropout(0.02) # Original dropout for EncoderLayer, keep fixed
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.norm2 = LayerNorm(d_model=d_model)

    def forward(self, embed, src_mask):
        attn_output, attention_weights = self.attention(q=embed, k=embed, v=embed, mask=src_mask)
        attn_output = self.dropout(attn_output)
        norm1_output = self.norm1(embed + attn_output)
        ffn_output = self.ffn(norm1_output)
        ffn_output = self.dropout(ffn_output)
        final_output = self.norm2(norm1_output + ffn_output)
        return final_output, attention_weights

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size=enc_voc_size, d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head) for _ in range(n_layers)])

    def forward(self, x_tokens, src_mask=None):
        x = self.emb(x_tokens)
        for layer in self.layers:
            x, attention = layer(x, src_mask) # Pass attention weights through, though not currently used downstream
        return x, attention

class siRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, seq_len):
        super().__init__()
        self.encoder = Encoder(vocab_size, seq_len, embedding_dim, embedding_dim * 4, n_head, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_tokens):
        encoded_sequence, attention = self.encoder(x_tokens)
        pooled_embedding = self.pool(encoded_sequence.permute(0, 2, 1)).squeeze(-1)
        return pooled_embedding, attention

class mRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, seq_len):
        super().__init__()
        # Note: The original code used embedding_dim * 2 for ffn_hidden here, keeping that
        self.encoder = Encoder(vocab_size, seq_len, embedding_dim, embedding_dim * 2, n_head, n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_tokens):
        encoded_sequence, attention = self.encoder(x_tokens)
        pooled_embedding = self.pool(encoded_sequence.permute(0, 2, 1)).squeeze(-1)
        return pooled_embedding, attention

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
                # Subsequent layers use the output size of the previous GNN layer
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

    def forward(self, x_dict, edge_index_dict, token_dict):
        # Get token sequences
        siRNA_tokens = token_dict['siRNA']       # shape: [num_siRNA_nodes, max_len]
        mRNA_tokens  = token_dict['mRNA']        # shape: [num_mRNA_nodes, max_len]

        # Encode
        siRNA_emb, _ = self.siRNA_encoder(siRNA_tokens)
        mRNA_emb, _  = self.mRNA_encoder(mRNA_tokens)

        # Concatenate embeddings with existing features (x_dict contains the 'other features')
        x_dict['siRNA'] = torch.cat([x_dict['siRNA'], siRNA_emb], dim=-1)
        x_dict['mRNA']  = torch.cat([x_dict['mRNA'],  mRNA_emb],  dim=-1)

        # GNN message passing
        x_dict_processed = x_dict
        # Run through GNN layers
        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict_processed = conv(x_dict_processed, edge_index_dict)
            for node_type in x_dict_processed:
                if node_type in bn_dict: # Only apply BatchNorm if it exists for this node type
                    if x_dict_processed[node_type].size(0) > 1 or not self.training: # Ensure batch norm works for batch size 1 during eval
                        x_dict_processed[node_type] = bn_dict[node_type](x_dict_processed[node_type])
                x_dict_processed[node_type] = F.leaky_relu(x_dict_processed[node_type])
                x_dict_processed[node_type] = dropout(x_dict_processed[node_type])

        # Final MLP or prediction head over interaction nodes
        return self.lin(x_dict_processed['interaction'])

    def l1_loss(self):
        # Applies L1 regularization to all model parameters
        return sum(p.abs().sum() for p in self.parameters())

print(f"Loaded Parameters: {params}")
print(f"Max siRNA Length for Padding: {MAX_SIRNA_LENGTH}")

# --- Data Loading and Preprocessing (Moved outside objective for efficiency) ---
# This part is complex and should ideally run only once per fold/split.
# For Optuna, we're focusing on a single split (Fold 0).

# Assuming we're only working on fold 0 for tuning
n = 0 # Fixed to the first split as requested for tuning

print(f"\n--- Preparing Data for Fold {n} ---")

# Read and preprocess data
split_dir = f"siRNA_split_datasets/split{n}/"
if not os.path.exists(split_dir):
    raise FileNotFoundError(f"Error: Directory '{split_dir}' not found. Please ensure your data splits are correctly organized.")

try:
    data_train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    data_dev_df = pd.read_csv(os.path.join(split_dir, "dev.csv"))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Error: Data file not found in '{split_dir}'. {e}")

data_train_df['split'] = 'train'
data_dev_df['split'] = 'dev'

# Substitute U to T for siRNA_seq to match mRNA_seq (DNA-like)
for df in [data_train_df, data_dev_df]:
    df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')

data_combined_df = pd.concat([data_train_df, data_dev_df], axis=0).reset_index(drop=True)

# Check vocabulary size
print("\n--- Checking Vocabulary for siRNA and mRNA Sequences ---")
sirna_chars = set(''.join(data_combined_df['siRNA_seq'].unique()))
mrna_chars = set(''.join(data_combined_df['mRNA_seq_RNA-FM'].unique()))
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
unique_sirna = data_combined_df[['siRNA', 'siRNA_seq']].drop_duplicates(subset='siRNA')
siRNA_tokens_list = [tokenize_sequence(seq, MAX_SIRNA_LENGTH) for seq in unique_sirna['siRNA_seq']]
siRNA_tokens_df = pd.DataFrame(siRNA_tokens_list, index=unique_sirna['siRNA'])

# 2. Tokenize mRNA sequences
unique_mrna = data_combined_df[['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset='mRNA')
mRNA_tokens_list = [tokenize_sequence(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']]
mRNA_tokens_df = pd.DataFrame(mRNA_tokens_list, index=unique_mrna['mRNA'])

# 3. Positional encoding for interactions
siRNA_pos_encoding = []
for idx, row in data_combined_df.iterrows():
    mrna_start_pos = max(0, int(row['pos']))
    siRNA_pos_encoding.append(utils1.get_pos_embedding_sequence(
        mrna_start_pos,
        len(row['siRNA_seq']), # Use actual siRNA length for positional encoding context
        MAX_SIRNA_LENGTH,
        params["dmodel"] # This should match embedding_dim if used as part of it
    ))
siRNA_pos_encoding_df = pd.DataFrame(siRNA_pos_encoding, index=data_combined_df['siRNA'] + '_' + data_combined_df['mRNA'])

# 4. Thermodynamics
siRNA_thermo_feat_list = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                          for seq in data_combined_df['siRNA_seq']]
siRNA_thermo_feat_df = pd.DataFrame(siRNA_thermo_feat_list).reset_index(drop=True)
temp_interaction_index = data_combined_df['siRNA'] + '_' + data_combined_df['mRNA']
siRNA_thermo_feat_df['index'] = temp_interaction_index
siRNA_thermo_feat_df = siRNA_thermo_feat_df.set_index('index')

# 5. Co-fold features
con_feat = pd.read_csv("siRNA_split_preprocess/full_con_matrix.txt", header=None, index_col=0)
con_feat = con_feat.reindex(siRNA_thermo_feat_df.index).fillna(0)

# 6. Self-fold features (siRNA)
siRNA_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
siRNA_sfold_feat = siRNA_sfold_feat.reindex(siRNA_tokens_df.index).fillna(0)

# 7. Self-fold features (mRNA)
mRNA_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
mRNA_sfold_feat = mRNA_sfold_feat.reindex(mRNA_tokens_df.index).fillna(0)

# 8. GC percentage (variable length robust)
siRNA_GC = pd.DataFrame([utils1.countGC(seq) for seq in data_combined_df['siRNA_seq']], index=list(data_combined_df['siRNA']))
mRNA_GC = pd.DataFrame([utils1.countGC(seq) for seq in unique_mrna['mRNA_seq_RNA-FM']], index=list(unique_mrna['mRNA']))

# 9. K-mers (All now return fixed-size lists)
siRNA_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data_combined_df['siRNA_seq']])
siRNA_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data_combined_df['siRNA_seq']])
siRNA_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_combined_df['siRNA_seq']])
siRNA_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_combined_df['siRNA_seq']])
siRNA_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_combined_df['siRNA_seq']])
siRNA_k_mers = pd.concat([siRNA_1_mer, siRNA_2_mers, siRNA_3_mers, siRNA_4_mers, siRNA_5_mers], axis=1)
siRNA_k_mers.index = data_combined_df['siRNA']

mRNA_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
mRNA_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
mRNA_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
mRNA_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
mRNA_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mrna['mRNA_seq_RNA-FM']])
mRNA_k_mers = pd.concat([mRNA_1_mer, mRNA_2_mers, mRNA_3_mers, mRNA_4_mers, mRNA_5_mers], axis=1)
mRNA_k_mers.index = unique_mrna['mRNA']

# 10. siRNA rules codes (PADDED to 19*3)
SIRNA_RULES_LENGTH = 19
siRNA_pos_scores = []
for seq in data_combined_df['siRNA_seq']:
    siRNA_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
siRNA_pos_scores = pd.DataFrame(siRNA_pos_scores, index=list(data_combined_df['siRNA']))

print("\n--- Assembling GNN Node Features ---")

# Create heterogeneous graph data structure
data_hetero = HeteroData()

# Add node features:
# siRNA and mRNA nodes will ONLY store tokenized sequences for the encoders.
# Other features (self-fold) will be stored separately as global tensors,
# and looked up during batch processing.
# Concatenate all siRNA features
siRNA_all_feats = pd.concat([
    siRNA_sfold_feat,
    siRNA_GC,
    siRNA_k_mers,
    siRNA_pos_scores
], axis=1)
# Ensure to convert to numpy first, then to_tensor for consistency
data_hetero['siRNA'].x = torch.tensor(siRNA_all_feats.values, dtype=torch.float)

# Concatenate all mRNA features
mRNA_all_feats = pd.concat([
    mRNA_sfold_feat,
    mRNA_GC,
    mRNA_k_mers
], axis=1)
data_hetero['mRNA'].x = torch.tensor(mRNA_all_feats.values, dtype=torch.float)

data_hetero['interaction'].x = torch.tensor(
    pd.concat([siRNA_thermo_feat_df, con_feat, siRNA_pos_encoding_df], axis=1).values,
    dtype=torch.float
)

# Ensure index order of tokens matches the node features
siRNA_tokens_aligned = siRNA_tokens_df.loc[siRNA_all_feats.index].values
mRNA_tokens_aligned = mRNA_tokens_df.loc[mRNA_all_feats.index].values

data_hetero['siRNA'].tokens = torch.tensor(siRNA_tokens_aligned, dtype=torch.long)
data_hetero['mRNA'].tokens = torch.tensor(mRNA_tokens_aligned, dtype=torch.long)

# Create edge indices
siRNA_name_to_idx = {name: i for i, name in enumerate(siRNA_tokens_df.index)}
mRNA_name_to_idx = {name: i for i, name in enumerate(mRNA_tokens_df.index)}
interaction_name_to_idx = {name: i for i, name in enumerate(siRNA_thermo_feat_df.index)} # Use thermo feat df index

edge_index_siRNA_to_interaction = []
edge_index_mRNA_to_interaction = []

for _, row in data_combined_df.iterrows():
    siRNA_name = row['siRNA']
    mRNA_name = row['mRNA']
    interaction_name = f"{siRNA_name}_{mRNA_name}"
    if (siRNA_name in siRNA_name_to_idx and
        mRNA_name in mRNA_name_to_idx and
        interaction_name in interaction_name_to_idx):
        edge_index_siRNA_to_interaction.append([siRNA_name_to_idx[siRNA_name], interaction_name_to_idx[interaction_name]])
        edge_index_mRNA_to_interaction.append([mRNA_name_to_idx[mRNA_name], interaction_name_to_idx[interaction_name]])
    else:
        # This warning happens if some entries in the combined data_df are not in the feature dfs,
        # which shouldn't happen if feature DFs are built from unique_siRNA/mRNA.
        # Keeping original warning for fidelity.
        # print(f"Warning: Skipping edge for {interaction_name} due to missing node in feature dataframes.")
        pass # Suppress repeated warnings during tuning for cleaner output

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
train_idx_list = []
dev_idx_list = []

for _, row in data_train_df.iterrows(): # Use original train_df
    interaction_name = f"{row['siRNA']}_{row['mRNA']}"
    if interaction_name in interaction_name_to_idx:
        train_idx_list.append(interaction_name_to_idx[interaction_name])

for _, row in data_dev_df.iterrows(): # Use original dev_df
    interaction_name = f"{row['siRNA']}_{row['mRNA']}"
    if interaction_name in interaction_name_to_idx:
        dev_idx_list.append(interaction_name_to_idx[interaction_name])

train_idx = torch.tensor(train_idx_list, dtype=torch.long)
dev_idx = torch.tensor(dev_idx_list, dtype=torch.long)

# Create labels tensor
labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
for _, row in data_combined_df.iterrows(): # Use combined data for all labels
    interaction_name = f"{row['siRNA']}_{row['mRNA']}"
    if interaction_name in interaction_name_to_idx:
        labels[interaction_name_to_idx[interaction_name]] = row['efficacy']
data_hetero['interaction'].y = labels

print(f"\n--- PyG HeteroData Label Shape ---")
print(f"data_hetero['interaction'].y shape: {data_hetero['interaction'].y.shape}")

# Determine feature dimensions for the HeteroSAGE model
siRNA_other_feature_dim = siRNA_all_feats.shape[1]
mRNA_other_feature_dim = mRNA_all_feats.shape[1]
interaction_feature_dim = data_hetero['interaction'].x.shape[1]


# --- Optuna Objective Function ---
def objective(trial):
    # --- 1. Hyperparameter Suggestions ---
    # GNN Layer Sizes & Number of Layers
    num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 4)
    gnn_hidden_channels = trial.suggest_categorical("gnn_hidden_channels", [32, 64, 128, 256])
    layer_sizes = [gnn_hidden_channels] * num_gnn_layers

    # Neighbor Sampling Strategy
    hop_samples = []
    for i in range(num_gnn_layers):
        hop_samples.append(trial.suggest_int(f"hop{i+1}_samples", 5, 20))

    # Dropout Rate (GNN layers)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.7, step=0.05)

    # Learning Rate
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Weight Decay (L2 regularization)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # Number of Attention Heads for Transformer Encoders (n_head)
    # dmodel is fixed at 12 as per your instruction in the outer params
    # n_head must be a divisor of dmodel.
    # Original params.json used 256 for dmodel, but your local override sets 12.
    # The valid divisors for dmodel=12 are [1, 2, 3, 4, 6, 12].
    # Let's offer a reasonable subset of these.
    n_head = trial.suggest_categorical("n_head", [1, 2, 3, 4, 6])


    # --- 2. Model Initialization with suggested HPs ---
    # Initialize NeighborLoaders with suggested hop_samples
    # batch_size is fixed at 16 as per your instruction
    train_loader = NeighborLoader(
        data_hetero,
        num_neighbors=hop_samples, # Use tuned hop_samples
        batch_size=params["batch_size"], # Fixed batch_size
        input_nodes=('interaction', train_idx),
        shuffle=True,
        subgraph_type='induced',
        filter_per_worker=params["filter_per_worker"],
        num_workers=params["num_workers"]
    )

    val_loader = NeighborLoader(
        data_hetero,
        num_neighbors=hop_samples, # Use tuned hop_samples
        batch_size=params["batch_size"], # Fixed batch_size
        input_nodes=('interaction', dev_idx),
        shuffle=False,
        subgraph_type='induced',
        filter_per_worker=params["filter_per_worker"],
        num_workers=params["num_workers"]
    )

    model = HeteroSAGE(
        vocab_size=vocab_size,
        embedding_dim=params["embedding_dim"], # Fixed embedding_dim
        n_head=n_head, # Tuned n_head
        n_layers=params["n_layers"], # Fixed n_layers (default 4)
        layer_sizes=layer_sizes, # Tuned GNN layer sizes
        out_channels=1,
        metadata=data_hetero.metadata(),
        sirna_seq_len=MAX_SIRNA_LENGTH,
        mrna_seq_len=MAX_MRNA_LENGTH,
        sirna_other_feature_dim=siRNA_other_feature_dim,
        mrna_other_feature_dim=mRNA_other_feature_dim,
        interaction_feature_dim=interaction_feature_dim,
        dropout_rate=dropout_rate # Tuned dropout
    ).to(device)

    # --- 3. Optimizer and Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params["scheduler_factor"], patience=params["scheduler_patience"], min_lr=params["min_lr"], verbose=False) # Set verbose to False for cleaner Optuna output
    criterion = nn.MSELoss() # Fixed to MSE as per your params["loss"]="mse"

    # --- 4. Training and Validation Loops ---
    def train_epoch(current_model, current_optimizer, current_loader, current_criterion, current_scaler, l1_lambda):
        current_model.train()
        total_loss = 0
        for batch in current_loader:
            batch = batch.to(device)
            current_optimizer.zero_grad()
            with autocast():
                token_dict = {
                    'siRNA': batch['siRNA'].tokens,
                    'mRNA': batch['mRNA'].tokens
                }
                out = current_model(batch.x_dict, batch.edge_index_dict, token_dict=token_dict)
                loss = current_criterion(out.squeeze(-1), batch['interaction'].y)
                if l1_lambda > 0:
                    loss += l1_lambda * current_model.l1_loss()
            current_scaler.scale(loss).backward()
            current_scaler.unscale_(current_optimizer)
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=params["clip_grad_norm"])
            current_scaler.step(current_optimizer)
            current_scaler.update()
            total_loss += loss.item()
        return total_loss / len(current_loader)

    @torch.no_grad()
    def validate_epoch(current_model, current_loader, current_criterion):
        current_model.eval()
        preds, truths = [], []
        total_val_loss = 0
        for batch in current_loader:
            batch = batch.to(device)
            token_dict = {
                'siRNA': batch['siRNA'].tokens,
                'mRNA': batch['mRNA'].tokens
            }
            with autocast():
                out = current_model(batch.x_dict, batch.edge_index_dict, token_dict=token_dict)
                loss = current_criterion(out.squeeze(-1), batch['interaction'].y)
            preds.append(out.cpu())
            truths.append(batch['interaction'].y.cpu())
            total_val_loss += loss.item()
        val_loss = F.mse_loss(torch.cat(preds).squeeze(-1), torch.cat(truths)).item()
        return val_loss # Return the calculated MSE for Optuna to minimize

    # Main Training Loop for the trial
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = GradScaler()

    for epoch in range(params["epochs"]):
        train_loss = train_epoch(model, optimizer, train_loader, criterion, scaler, params.get("l1_lambda", 0.0))
        val_loss = validate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        # Optuna Pruning: Report intermediate value to the trial.
        # This allows Optuna to stop unpromising trials early.
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Optional: Save best model state dict if needed, but not strictly required for Optuna itself
            # torch.save(model.state_dict(), f'best_model_trial_{trial.number}.pt')
            
        else:
            patience_counter += 1
            if patience_counter >= params["early_stopping_patience"]:
                # print(f"Early stopping triggered after {epoch + 1} epochs for trial {trial.number}.")
                break
    
    # Clean up GPU memory for the next trial
    del model
    del optimizer
    del scaler
    torch.cuda.empty_cache()

    return best_val_loss # Optuna will minimize this value

# --- Optuna Study Creation and Optimization ---
# To create new memory for each run (or load if it exists), use a file-based storage.
# This prevents conflicts and allows you to resume studies.
# Make sure the directory for the DB exists
os.makedirs('optuna_studies', exist_ok=True)
storage_path = 'sqlite:///optuna_studies/siRNA_gnn_tuning_study_new.db' # Use a new unique name

# Create or load the study
# Use load_if_exists=True to prevent errors if the study already exists from a previous run
# and to continue adding trials to it.
# If you truly want a *brand new* study every time, delete the .db file first.
print(f"\n--- Initializing Optuna Study (Storage: {storage_path}) ---")
study = optuna.create_study(
    direction="minimize",
    study_name="siRNA_GNN_Hyperparameter_Tuning_Fold",
    storage=storage_path,
    load_if_exists=False,
    sampler=optuna.samplers.TPESampler(seed=42) # For reproducibility
)

# Run the optimization
# You can set n_trials to limit the number of trials, or timeout for time-based limit
print("\n--- Starting Optuna Optimization ---")
study.optimize(objective, n_trials=25, gc_after_trial=True) # Run 50 trials, clear GPU memory after each

print("\n--- Optuna Optimization Complete ---")
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Optional: You can load the best model and evaluate it on the dev set one last time
# This part is outside the Optuna objective as it's for final evaluation, not tuning.
print("\n--- Final Evaluation with Best Trial Parameters ---")
best_params = trial.params

# Re-initialize loaders with best hop_samples
best_hop_samples = []
for i in range(best_params["num_gnn_layers"]):
    best_hop_samples.append(best_params[f"hop{i+1}_samples"])

best_train_loader = NeighborLoader(
    data_hetero,
    num_neighbors=best_hop_samples,
    batch_size=params["batch_size"],
    input_nodes=('interaction', train_idx),
    shuffle=True,
    subgraph_type='induced',
    filter_per_worker=params["filter_per_worker"],
    num_workers=params["num_workers"]
)

best_val_loader = NeighborLoader(
    data_hetero,
    num_neighbors=best_hop_samples,
    batch_size=params["batch_size"],
    input_nodes=('interaction', dev_idx),
    shuffle=False,
    subgraph_type='induced',
    filter_per_worker=params["filter_per_worker"],
    num_workers=params["num_workers"]
)

# Re-initialize model with best parameters
best_model = HeteroSAGE(
    vocab_size=vocab_size,
    embedding_dim=params["embedding_dim"],
    n_head=best_params["n_head"],
    n_layers=params["n_layers"],
    layer_sizes=[best_params["gnn_hidden_channels"]] * best_params["num_gnn_layers"],
    out_channels=1,
    metadata=data_hetero.metadata(),
    sirna_seq_len=MAX_SIRNA_LENGTH,
    mrna_seq_len=MAX_MRNA_LENGTH,
    sirna_other_feature_dim=siRNA_other_feature_dim,
    mrna_other_feature_dim=mRNA_other_feature_dim,
    interaction_feature_dim=interaction_feature_dim,
    dropout_rate=best_params["dropout"]
).to(device)

# Load best state dict if you saved it (currently commented out in objective)
# If you want to load the actual best model, you'd need to uncomment the torch.save line
# in the objective and then load it here:
# best_model.load_state_dict(torch.load(f'best_model_trial_{trial.number}.pt'))

# Or, simply train a new model from scratch with the best parameters
# This is often done if you don't save intermediate models during tuning
# Train for a few epochs with the best configuration (similar to your original loop)
best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
best_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(best_optimizer, mode='min', factor=params["scheduler_factor"], patience=params["scheduler_patience"], min_lr=params["min_lr"], verbose=True)
best_criterion = nn.MSELoss()

best_val_loss = float('inf')
best_patience_counter = 0
final_scaler = GradScaler() # New scaler for final training

print("\n--- Retraining best model for final evaluation ---")
for epoch in range(params["epochs"]):
    train_loss = train_epoch(best_model, best_optimizer, best_train_loader, best_criterion, final_scaler, params.get("l1_lambda", 0.0))
    val_loss = validate_epoch(best_model, best_val_loader, best_criterion)
    best_scheduler.step(val_loss)
    if epoch%10==0:
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_patience_counter = 0
        torch.save(best_model.state_dict(), f'final_best_model_fold{n}.pt') # Save the truly best model for this fold
    else:
        best_patience_counter += 1
        if best_patience_counter >= params["early_stopping_patience"]:
            print(f"Early stopping triggered for final best model after {epoch + 1} epochs.")
            break
    print(f'Final Best Model Training - Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Final evaluation on the validation set using the best model
final_preds, final_truths = validate_epoch(best_model, best_val_loader, best_criterion)
final_mse = F.mse_loss(final_preds.squeeze(-1), final_truths).item()
final_pcc, _ = scipy.stats.pearsonr(final_preds.squeeze(-1).numpy(), final_truths.numpy())

print(f"\n--- Final Best Model Performance on Validation Set (Fold {n}) ---")
print(f"Final Validation MSE: {final_mse:.4f}")
print(f"Final Validation PCC: {final_pcc:.4f}")