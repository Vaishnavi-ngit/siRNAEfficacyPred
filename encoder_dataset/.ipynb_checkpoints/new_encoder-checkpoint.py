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
from sklearn.metrics import mean_squared_error
import scipy.stats
import json
import math
from torch.cuda.amp import GradScaler, autocast # Import for mixed precision training

torch.cuda.manual_seed_all(42)
import BugUtils as utils1

# --- Load parameters ---
with open("siRNA_param_pytorch.json", 'r') as f:
    params = json.load(f)

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

# Add default values for missing parameters
params.setdefault("embedding_dim", 128)  # Default embedding dimension
params.setdefault("n_head", 8)           # Default number of attention heads
params.setdefault("n_layers", 2)         # Default number of transformer layers
params.setdefault("dmodel", 128) # Ensure dmodel is set for positional encoding in utils1

# Define device here, at the top level, so it's accessible globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.dropout = nn.Dropout(0.02)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
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
        self.dropout = nn.Dropout(0.02)
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
            x, attention = layer(x, src_mask)
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
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Calculate input dimensions for the first HeteroConv layer
        # Sequence embeddings are embedding_dim
        # Other features are sirna_other_feature_dim and mrna_other_feature_dim
        sirna_initial_dim = embedding_dim + sirna_other_feature_dim
        mrna_initial_dim = embedding_dim + mrna_other_feature_dim

        for i in range(len(layer_sizes)):
            if i == 0:
                in_channels_dict = {
                    'siRNA': sirna_initial_dim,
                    'mRNA': mrna_initial_dim,
                    'interaction': interaction_feature_dim
                }
            else:
                in_channels_dict = {node_type: layer_sizes[i-1] for node_type in self.node_types}
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

    def forward(self, x_dict, edge_index_dict, sirna_other_features, mrna_other_features):
        # x_dict['siRNA'] and x_dict['mRNA'] now contain only tokenized sequences
        sirna_tokens = x_dict['siRNA'].long()
        mrna_tokens = x_dict['mRNA'].long()

        sirna_emb, _ = self.siRNA_encoder(sirna_tokens)
        mrna_emb, _ = self.mRNA_encoder(mrna_tokens)

        # Concatenate sequence embeddings with other features
        sirna_node_features = torch.cat([sirna_emb, sirna_other_features], dim=-1)
        mrna_node_features = torch.cat([mrna_emb, mrna_other_features], dim=-1)

        x_dict_processed = {
            'siRNA': sirna_node_features,
            'mRNA': mrna_node_features,
            'interaction': x_dict['interaction'] # Interaction features are already combined
        }

        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict_processed = conv(x_dict_processed, edge_index_dict)
            for node_type in x_dict_processed:
                if node_type in bn_dict:
                    if x_dict_processed[node_type].size(0) > 1 or not self.training:
                        x_dict_processed[node_type] = bn_dict[node_type](x_dict_processed[node_type])
                x_dict_processed[node_type] = F.leaky_relu(x_dict_processed[node_type])
                x_dict_processed[node_type] = dropout(x_dict_processed[node_type])

        return self.lin(x_dict_processed['interaction'])

    def l1_loss(self):
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
    except FileNotFoundError as e:
        print(f"Error: Data file not found in '{split_dir}'. {e}")
        print("Skipping fold processing.")
        continue

    data_train['split'] = 'train'
    data_dev['split'] = 'dev'

    # Substitute U to T for siRNA_seq to match mRNA_seq (DNA-like)
    for df in [data_train, data_dev]:
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')

    data = pd.concat([data_train, data_dev], axis=0).reset_index(drop=True)

    # Check vocabulary size
    print("\n--- Checking Vocabulary for siRNA and mRNA Sequences ---")
    sirna_chars = set(''.join(data['siRNA_seq'].unique()))
    mrna_chars = set(''.join(data['mRNA_seq_RNA-FM'].unique()))
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
    con_feat = pd.read_csv("siRNA_split_preprocess/full_con_matrix.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat_df.index).fillna(0)

    # 6. Self-fold features (siRNA)
    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_tokens_df.index).fillna(0)

    # 7. Self-fold features (mRNA)
    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_tokens_df.index).fillna(0)

    print("\n--- Assembling GNN Node Features ---")

    # Create heterogeneous graph data structure
    data_hetero = HeteroData()

    # Add node features:
    # siRNA and mRNA nodes will ONLY store tokenized sequences for the encoders.
    # Other features (self-fold) will be stored separately as global tensors,
    # and looked up during batch processing.
    data_hetero['siRNA'].x = torch.tensor(sirna_tokens_df.values, dtype=torch.long)
    data_hetero['mRNA'].x = torch.tensor(mrna_tokens_df.values, dtype=torch.long)
    data_hetero['interaction'].x = torch.tensor(
        pd.concat([sirna_thermo_feat_df, con_feat, sirna_pos_encoding_df], axis=1).values,
        dtype=torch.float
    )

    # Store self-fold features globally for lookup by node ID in the model's forward pass
    # Ensure indices align for proper lookup.
    # Move these global tensors to the correct device at initialization
    sirna_sfold_tensor = torch.tensor(sirna_sfold_feat.values, dtype=torch.float).to(device)
    mrna_sfold_tensor = torch.tensor(mrna_sfold_feat.values, dtype=torch.float).to(device)


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
    train_idx_list = []
    dev_idx_list = []

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
    train_loader = NeighborLoader(
        data_hetero,
        num_neighbors=params["hop_samples"],
        batch_size=params["batch_size"],
        input_nodes=('interaction', train_idx),
        shuffle=True,
        subgraph_type='induced',
        filter_per_worker=params["filter_per_worker"],
        num_workers=params["num_workers"]
    )

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
    sirna_other_feature_dim = sirna_sfold_feat.shape[1]
    mrna_other_feature_dim = mrna_sfold_feat.shape[1]
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

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)
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

    # --- Training Function ---
    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            batch_siRNA_sfold = sirna_sfold_tensor[batch['siRNA'].n_id]
            batch_mRNA_sfold = mrna_sfold_tensor[batch['mRNA'].n_id]
            
            out = model(batch.x_dict, batch.edge_index_dict, batch_siRNA_sfold, batch_mRNA_sfold)

            # Ensure batch['interaction'].y has the correct shape for loss calculation
            # It should be batch.y but accessing via batch['interaction'].y is specific to HeteroData
            loss = criterion(out.squeeze(-1), batch['interaction'].y)

            l1_lambda = params.get("l1_lambda", 1e-5)
            loss += l1_lambda * model.l1_loss()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            batch = batch.to(device)
            # Lookup self-fold features for the current batch's nodes
            # These tensors are now correctly moved to device globally
            batch_siRNA_sfold = sirna_sfold_tensor[batch['siRNA'].n_id]
            batch_mRNA_sfold = mrna_sfold_tensor[batch['mRNA'].n_id]

            #with autocast():  # Enable mixed precision
            out = model(batch.x_dict, batch.edge_index_dict, batch_siRNA_sfold, batch_mRNA_sfold)
            preds.append(out.cpu())
            truths.append(batch['interaction'].y.cpu())
        return torch.cat(preds), torch.cat(truths)

    # Main Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(params["epochs"]):
        train_loss = train()
        val_pred, val_true = validate(val_loader)
        val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_fold{n}_enc.pt')
        else:
            patience_counter += 1
            if patience_counter >= params["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("\n--- Training and Validation Complete Across All Folds ---")
print("Model checkpoints saved based on best validation loss for each fold.")