# Assuming this is your Notting.py file contents

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData 
from torch_geometric.loader import NeighborLoader 
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch.nn import Dropout, LayerNorm
import math

# --- Define Token IDs as constants for consistency ---
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 5 # Used for CLS token


# --- Tokenization Function (with CLS token prepended) ---
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
    def __init__(self, vocab_size, d_model, n_head, n_layers, max_len, dropout, additional_feature_dim=0):
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
            nn.Linear(d_model * 2 + additional_feature_dim, d_model),
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

        if additional_features is not None:
            pooled = torch.cat([pooled, additional_features], dim=-1)

        return self.output_proj(pooled)

# --- HeteroGNN Class ---
class HeteroGNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, gat_heads, layer_sizes,
                     out_channels, metadata, sirna_seq_len, mrna_seq_len,
                     sirna_other_feature_dim, mrna_other_feature_dim,
                     sirna_seq_feat_dim, mrna_seq_feat_dim,
                     interaction_feature_dim, dropout_rate):
        
        super().__init__()

        self.siRNA_encoder = TransformerEncoder(
            vocab_size, embedding_dim, n_head, n_layers, sirna_seq_len, dropout_rate,
            additional_feature_dim=sirna_seq_feat_dim
        )
        self.mRNA_encoder = TransformerEncoder(
            vocab_size, embedding_dim, n_head, n_layers, mrna_seq_len, dropout_rate,
            additional_feature_dim=mrna_seq_feat_dim
        )

        self.feature_norms = nn.ModuleDict({
            'siRNA': LayerNorm(sirna_other_feature_dim),
            'mRNA': LayerNorm(mrna_other_feature_dim),
            'interaction': LayerNorm(interaction_feature_dim)
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_projs = nn.ModuleDict()

        sirna_initial_gnn_dim = sirna_other_feature_dim + embedding_dim
        mrna_initial_gnn_dim = mrna_other_feature_dim + embedding_dim
        
        for i, size in enumerate(layer_sizes):
            if i == 0:
                in_channels = {
                    'siRNA': sirna_initial_gnn_dim,
                    'mRNA': mrna_initial_gnn_dim,
                    'interaction': interaction_feature_dim,
                }
            else:
                in_channels = {key: layer_sizes[i-1] for key in metadata[0]}
            
            conv = HeteroConv({
                edge_type: GATv2Conv(
                    in_channels=(in_channels[edge_type[0]], in_channels[edge_type[2]]),
                    out_channels=size // gat_heads,
                    heads=gat_heads,
                    add_self_loops=False, # <-- FIXED: Changed back to False
                    dropout=dropout_rate,
                    bias=False
                )
                for edge_type in metadata[1]
            }, aggr='mean')

            self.convs.append(conv)
            
            self.norms.append(nn.ModuleDict({
                node_type: LayerNorm(size) for node_type in metadata[0]
            }))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            for node_type in metadata[0]:
                if i == 0:
                    residual_input_dim = in_channels[node_type]
                else:
                    residual_input_dim = layer_sizes[i-1]

                output_dim = size

                if residual_input_dim != output_dim:
                    proj_key = f"{node_type}_layer_{i}"
                    self.residual_projs[proj_key] = nn.Linear(residual_input_dim, output_dim, bias=False)


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

        # 2. Sequence encoding with additional features
        siRNA_emb = self.siRNA_encoder(token_dict['siRNA'], seq_feat_dict['siRNA'])
        x_dict['siRNA'] = torch.cat([x_dict['siRNA'], siRNA_emb], dim=-1)

        mRNA_emb = self.mRNA_encoder(token_dict['mRNA'], seq_feat_dict['mRNA'])
        x_dict['mRNA'] = torch.cat([x_dict['mRNA'], mRNA_emb], dim=-1)

        # 3. Graph convolution with residual connections and post-normalization/activation
        for i, (conv, norm_dict, dropout) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x_residual = {k: v.clone() for k, v in x_dict.items()}
            
            x_dict = conv(x_dict, edge_index_dict)

            for node_type, x in x_dict.items():
                residual = x_residual[node_type]
                
                proj_key = f"{node_type}_layer_{i}"
                if proj_key in self.residual_projs:
                    residual = self.residual_projs[proj_key](residual)
                
                x = x + residual
                
                x = norm_dict[node_type](x) 
                
                x = F.gelu(x)
                x = dropout(x)
                x_dict[node_type] = x

        # 4. Final prediction head
        return self.prediction_head(x_dict['interaction'])