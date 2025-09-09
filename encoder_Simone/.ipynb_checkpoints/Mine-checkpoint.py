import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" # Keeping it as is from your script, ensures GPU visibility
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
# IMPORTANT: Model imports must match your *trained* model's architecture
from torch_geometric.nn import HeteroConv, SAGEConv, Linear # For HeteroGNN with SAGEConv
from torch.nn import BatchNorm1d, Dropout
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler # Keeping StandardScaler as it was in your training script for feature creation
import scipy.stats
import json
import math
import re # Not directly used in the provided blocks, but keeping it as it was in the original
# import joblib # ### --- MODIFICATION: REMOVED - Not saving/loading scalers --- ###
from torch.cuda.amp import GradScaler, autocast

# --- CRITICAL: Import the custom Transformer components from the training script ---
# These classes must be defined here or imported from 'largeDataUtils' if they are external.
# I'm including them directly as they were in your training script.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
        seq_len = x.size(1)
        return self.pe[:seq_len, :].to(x.device).unsqueeze(0) + x

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
        return tok_emb + self.pos_emb(tok_emb)

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
    def __init__(self, d_model, hidden, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

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
    def __init__(self, d_model, ffn_hidden, n_head, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, dropout_rate=dropout_rate)
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
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, dropout_rate):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size=enc_voc_size, d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, dropout_rate=dropout_rate) for _ in range(n_layers)])

    def forward(self, x_tokens, src_mask=None):
        x = self.emb(x_tokens)
        for layer in self.layers:
            x, attention = layer(x, src_mask)
        return x, attention

class siRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, seq_len, dropout_rate):
        super().__init__()
        self.encoder = Encoder(vocab_size, seq_len, embedding_dim, embedding_dim * 4, n_head, n_layers, dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_tokens):
        encoded_sequence, attention = self.encoder(x_tokens)
        pooled_embedding = self.pool(encoded_sequence.permute(0, 2, 1)).squeeze(-1)
        return pooled_embedding, attention

class mRNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, seq_len, dropout_rate):
        super().__init__()
        self.encoder = Encoder(vocab_size, seq_len, embedding_dim, embedding_dim * 2, n_head, n_layers, dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_tokens):
        encoded_sequence, attention = self.encoder(x_tokens)
        pooled_embedding = self.pool(encoded_sequence.permute(0, 2, 1)).squeeze(-1)
        return pooled_embedding, attention

class HeteroSAGE(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, n_layers, layer_sizes, out_channels, metadata, sirna_seq_len, mrna_seq_len, sirna_other_feature_dim, mrna_other_feature_dim, interaction_feature_dim, dropout_rate=0.5):
        super().__init__()
        self.siRNA_encoder = siRNA_Encoder(vocab_size, embedding_dim, n_head, n_layers, seq_len=sirna_seq_len, dropout_rate=dropout_rate)
        self.mRNA_encoder = mRNA_Encoder(vocab_size, embedding_dim, n_head, n_layers, seq_len=mrna_seq_len, dropout_rate=dropout_rate)
        
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        sirna_input_dim = sirna_other_feature_dim + embedding_dim
        mrna_input_dim  = mrna_other_feature_dim + embedding_dim
        interaction_input_dim = interaction_feature_dim

        for i in range(len(layer_sizes)):
            if i == 0:
                in_channels_dict = {
                    'siRNA': sirna_input_dim,
                    'mRNA': mrna_input_dim,
                    'interaction': interaction_input_dim,
                }
            else:
                in_channels_dict = {node_type: layer_sizes[i - 1] for node_type in self.node_types}

            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels_dict['siRNA'], in_channels_dict['interaction']), out_channels_i), # ### MODIFIED: SAGEConv ###
                ('mRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels_dict['mRNA'], in_channels_dict['interaction']), out_channels_i), # ### MODIFIED: SAGEConv ###
                ('interaction', 'rev_interacts_with', 'siRNA'): SAGEConv((in_channels_dict['interaction'], in_channels_dict['siRNA']), out_channels_i), # ### MODIFIED: SAGEConv ###
                ('interaction', 'rev_interacts_with', 'mRNA'): SAGEConv((in_channels_dict['interaction'], in_channels_dict['mRNA']), out_channels_i), # ### MODIFIED: SAGEConv ###
            }, aggr='mean')
            self.convs.append(conv)

            self.bns.append(nn.ModuleDict({
                node_type: BatchNorm1d(out_channels_i) for node_type in self.node_types
            }))
            self.dropouts.append(Dropout(dropout_rate))

        self.lin = Linear(layer_sizes[-1], out_channels)

    def forward(self, x_dict, edge_index_dict, token_dict):
        siRNA_tokens = token_dict['siRNA']
        mRNA_tokens  = token_dict['mRNA']

        siRNA_emb, _ = self.siRNA_encoder(siRNA_tokens)
        mRNA_emb, _  = self.mRNA_encoder(mRNA_tokens)

        x_dict['siRNA'] = torch.cat([x_dict['siRNA'], siRNA_emb], dim=-1)
        x_dict['mRNA']  = torch.cat([x_dict['mRNA'],  mRNA_emb],  dim=-1)

        x_dict_processed = x_dict
        for conv, bn_dict, dropout in zip(self.convs, self.bns, self.dropouts):
            x_dict_processed = conv(x_dict_processed, edge_index_dict)
            for node_type in x_dict_processed:
                if node_type in bn_dict:
                    if x_dict_processed[node_type].size(0) > 1 or not self.training:
                        x_dict_processed[node_type] = bn_dict[node_type](x_dict_processed[node_type])
                x_dict_processed[node_type] = F.gelu(x_dict_processed[node_type]) # ### MODIFIED: GELU (as in your last training script) ###
                x_dict_processed[node_type] = dropout(x_dict_processed[node_type])

        return self.lin(x_dict_processed['interaction'])


# --- Global Parameters and Device Setup (Copied as-is from your training script's context) ---
try:
    with open("param.json", 'r') as f: # ### MODIFIED: param.json from siRNA_param_pytorch.json ###
        params = json.load(f)
except FileNotFoundError:
    print("Error: param.json not found. Please create it.") # ### MODIFIED: param.json from siRNA_param_pytorch.json ###
    exit()

# Define constants from parameters (re-setting defaults to ensure all exist)
MAX_SIRNA_LENGTH = params.setdefault("sirna_length", 19)
MAX_MRNA_LENGTH = params.setdefault("max_mrna_len", 128)
params.setdefault("embedding_dim", 256)
params.setdefault("dmodel", params["embedding_dim"])
params.setdefault("n_head", 4)
params.setdefault("n_layers", 4)
params.setdefault("gat_heads", 8) # Still present but unused if using SAGEConv
params.setdefault("hinsage_layer_sizes", [256, 128, 64])
params.setdefault("dropout", 0.2)
params.setdefault("batch_size", 64)
params.setdefault("hop_samples", [20, 15])
params.setdefault("clip_grad_norm", 0.5)
params.setdefault("lr", 0.001)
params.setdefault("weight_decay", 0.001)
params.setdefault("epochs", 300)
params.setdefault("early_stopping_patience", 30)
params.setdefault("loss", "mse")
params.setdefault("l1_lambda", 1e-6)
params.setdefault("huber_delta", 1.0)
params.setdefault("num_workers", 0)
params.setdefault("filter_per_worker", False) # Often True for large graphs with NeighborLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Utility Functions (Copied as-is from your training script's context) ---
# tokenize_sequence (defined above)
import Utils as utils1 # ### MODIFIED: Back to Utils as in your training script (instead of largeDataUtils) ###

# --- Helper Function for Evaluation (Adapted from your training validation logic) ---
@torch.no_grad()
def evaluate_model_for_test(model, loader, efficacy_threshold_for_auc):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        token_dict = {
            'siRNA': batch['siRNA'].tokens,
            'mRNA': batch['mRNA'].tokens
        }
        with autocast():
            out = model(batch.x_dict, batch.edge_index_dict, token_dict=token_dict).squeeze()
        preds.append(out.cpu())
        truths.append(batch['interaction'].y.cpu())

    test_preds = torch.cat(preds).numpy().flatten()
    test_true = torch.cat(truths).numpy().flatten()

    valid_mask = ~np.isnan(test_preds) & ~np.isnan(test_true)
    if not np.any(valid_mask):
        print("Warning: All predictions or true values are NaN for the test set. Skipping metrics.")
        return np.nan, np.nan, np.nan, np.nan

    test_preds = test_preds[valid_mask]
    test_true = test_true[valid_mask]

    pcc, _ = scipy.stats.pearsonr(test_true, test_preds)
    spcc, _ = scipy.stats.spearmanr(test_true, test_preds)
    mse = mean_squared_error(test_true, test_preds)
    
    bin_test_true = (test_true >= efficacy_threshold_for_auc).astype(int)
    if len(np.unique(bin_test_true)) > 1:
        auc = roc_auc_score(bin_test_true, test_preds)
    else:
        auc = np.nan

    return pcc, spcc, mse, auc


# --- Main Testing Logic ---
def main_test():
    print("Starting siRNA Efficacy Prediction Model TESTING (Simone Dataset)")
    print(f"Using device: {DEVICE}")
    print(f"Parameters: {params}")

    # ### --- MODIFICATION: Define Paths for Simone Test Dataset --- ###
    TEST_DATA_CSV = "../encoder_Simone/Simone_all_data.csv"
    SIRNA_SFOLD_PATH = "../encoder_Simone/Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt"
    MRNA_SFOLD_PATH = "../encoder_Simone/Simone_split_preprocess/self_mRNA_matrix.txt"
    CON_MATRIX_PATH = "../encoder_Simone/Simone_split_preprocess/con_matrix.txt"
    
    MODEL_DIR = "./" # Directory where your trained models are saved (e.g., best_model_fold_0.pt)
    # ### --- MODIFICATION: REMOVED SCALERS_DIR and joblib.load calls --- ###
    # SCALERS_DIR = "./scalers/" # REMOVED

    all_pcc, all_spcc, all_mse, all_auc = [], [], [], []

    # ### --- MODIFICATION: Test for ONE fold (n=0) based on typical saved models --- ###
    # If you have multiple models saved (e.g., best_model_fold_0.pt, best_model_fold_1.pt, etc.)
    # and want to average results over them, you can change NUM_FOLDS_TO_TEST.
    NUM_FOLDS_TO_TEST = 1 # Set to 1 to test only fold 0, or higher if you have multiple saved models
    
    for n in range(NUM_FOLDS_TO_TEST):
        print(f"\n{'='*40}\n--- Testing with Model from Fold {n} ---\n{'='*40}")

        # Step 1: Load Test Data
        try:
            data = pd.read_csv(TEST_DATA_CSV)
            print(f"Successfully loaded test data from: {TEST_DATA_CSV}")
        except FileNotFoundError:
            print(f"FATAL: Test data file not found at {TEST_DATA_CSV}. Exiting this fold.")
            continue
        except Exception as e:
            print(f"An error occurred while loading the test CSV file: {e}. Skipping this fold.")
            continue
        
        # Preprocessing on sequences (U to T, uppercase)
        data['siRNA_seq'] = data['siRNA_seq'].str.replace('U', 'T').str.upper()
        data['mRNA_seq_RNA-FM'] = data['mRNA_seq_RNA-FM'].str.upper()

        # Step 2: Feature Generation (Exact matching of training process)
        print("Generating features for test data...")
        
        vocab_size = len(tokenize_sequence('ACGTUNR-', 1)) + 1 # Calculate max possible vocab size + 0 for pad. Should be 9.
        
        unique_siRNA_test = data[['siRNA', 'siRNA_seq']].drop_duplicates(subset='siRNA')
        unique_mRNA_test = data[['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset='mRNA')

        sirna_tokens_list_test = [tokenize_sequence(seq, MAX_SIRNA_LENGTH) for seq in unique_siRNA_test['siRNA_seq']]
        sirna_tokens_df_test = pd.DataFrame(sirna_tokens_list_test, index=unique_siRNA_test['siRNA'])

        mrna_tokens_list_test = [tokenize_sequence(seq, MAX_MRNA_LENGTH) for seq in unique_mRNA_test['mRNA_seq_RNA-FM']]
        mrna_tokens_df_test = pd.DataFrame(mrna_tokens_list_test, index=unique_mRNA_test['mRNA'])

        # Load precomputed features using the specified Simone test paths
        try:
            con_feat_raw_from_file_test = pd.read_csv(CON_MATRIX_PATH, header=None, index_col=0)
            sirna_sfold_feat_raw_test = pd.read_csv(SIRNA_SFOLD_PATH, header=None, index_col=0)
            mrna_sfold_feat_raw_test = pd.read_csv(MRNA_SFOLD_PATH, header=None, index_col=0)
        except FileNotFoundError as e:
            print(f"Error loading precomputed feature file: {e}. Ensure test feature files exist.")
            print("Skipping this fold for testing.")
            continue
        except Exception as e:
            print(f"An error occurred loading feature files for fold {n}: {e}. Skipping fold.")
            continue

        interaction_index_map_test = data.apply(lambda row: f"{row['siRNA']}_{row['mRNA']}", axis=1)
        con_feat_test = con_feat_raw_from_file_test.reindex(interaction_index_map_test).fillna(0)

        sirna_sfold_feat_test = sirna_sfold_feat_raw_test.reindex(unique_siRNA_test.index).fillna(0)
        mrna_sfold_feat_test = mrna_sfold_feat_raw_test.reindex(unique_mRNA_test.index).fillna(0)

        sirna_pos_encoding_list_test = []
        for _, row in data.iterrows():
            mrna_start_pos = max(0, int(row['pos']))
            sirna_pos_encoding_list_test.append(utils1.get_pos_embedding_sequence(
                mrna_start_pos, len(row['siRNA_seq']), MAX_SIRNA_LENGTH, params["dmodel"]
            ))
        sirna_pos_encoding_df_test = pd.DataFrame(sirna_pos_encoding_list_test, index=interaction_index_map_test)

        sirna_thermo_feat_list_test = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH) for seq in data['siRNA_seq']]
        sirna_thermo_feat_df_test = pd.DataFrame(sirna_thermo_feat_list_test).set_index(interaction_index_map_test)

        sirna_GC_test = pd.DataFrame([utils1.countGC(seq) for seq in unique_siRNA_test['siRNA_seq']], index=unique_siRNA_test.index)
        mrna_GC_test = pd.DataFrame([utils1.countGC(seq) for seq in unique_mRNA_test['mRNA_seq_RNA-FM']], index=unique_mRNA_test.index)

        sirna_k_mers_test = pd.concat([
            pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in unique_siRNA_test['siRNA_seq']]),
            pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in unique_siRNA_test['siRNA_seq']]),
            pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in unique_siRNA_test['siRNA_seq']]),
            pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in unique_siRNA_test['siRNA_seq']]),
            pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in unique_siRNA_test['siRNA_seq']])
        ], axis=1, ignore_index=True)
        sirna_k_mers_test.index = unique_siRNA_test.index

        mrna_k_mers_test = pd.concat([
            pd.DataFrame([utils1.single_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mRNA_test['mRNA_seq_RNA-FM']]),
            pd.DataFrame([utils1.double_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mRNA_test['mRNA_seq_RNA-FM']]),
            pd.DataFrame([utils1.triple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mRNA_test['mRNA_seq_RNA-FM']]),
            pd.DataFrame([utils1.quadruple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mRNA_test['mRNA_seq_RNA-FM']]),
            pd.DataFrame([utils1.quintuple_freq(seq, MAX_MRNA_LENGTH) for seq in unique_mRNA_test['mRNA_seq_RNA-FM']])
        ], axis=1, ignore_index=True)
        mrna_k_mers_test.index = unique_mRNA_test.index

        SIRNA_RULES_LENGTH = params["sirna_length"]
        sirna_pos_scores_test = pd.DataFrame([utils1.rules_scores(seq, SIRNA_RULES_LENGTH) for seq in unique_siRNA_test['siRNA_seq']],
                                               index=unique_siRNA_test.index)

        # Concatenate ALL features for each node type (unscaled)
        sirna_all_feats_test_unscaled = pd.concat([
            sirna_sfold_feat_test, sirna_GC_test, sirna_k_mers_test, sirna_pos_scores_test
        ], axis=1).fillna(0)

        mrna_all_feats_test_unscaled = pd.concat([
            mrna_sfold_feat_test, mrna_GC_test, mrna_k_mers_test
        ], axis=1).fillna(0)

        interaction_all_feats_test_unscaled = pd.concat([
            sirna_thermo_feat_df_test, con_feat_test, sirna_pos_encoding_df_test
        ], axis=1).fillna(0)

        # ### --- CRITICAL MODIFICATION: Scaling logic is now internal to this script, as scalers are NOT loaded --- ###
        # This will fit a NEW StandardScaler on the test data.
        # THIS IS NOT IDEAL FOR REAL-WORLD INFERENCE where scaling should be consistent with training.
        # But, it directly follows your instruction that scalers are NOT saved/loaded.
        print("\nSCALER WARNING: Fitting StandardScaler on test data. This is not ideal for inference.")
        print("For robust inference, scalers must be saved during training and loaded here.")
        scaler_siRNA_x = StandardScaler()
        scaler_mRNA_x = StandardScaler()
        scaler_interaction_x = StandardScaler()
        
        sirna_all_feats_scaled = pd.DataFrame(
            scaler_siRNA_x.fit_transform(sirna_all_feats_test_unscaled), # ### MODIFIED: .fit_transform ###
            index=sirna_all_feats_test_unscaled.index,
            columns=sirna_all_feats_test_unscaled.columns
        )
        mrna_all_feats_scaled = pd.DataFrame(
            scaler_mRNA_x.fit_transform(mrna_all_feats_test_unscaled), # ### MODIFIED: .fit_transform ###
            index=mrna_all_feats_test_unscaled.index,
            columns=mrna_all_feats_test_unscaled.columns
        )
        interaction_all_feats_scaled = pd.DataFrame(
            scaler_interaction_x.fit_transform(interaction_all_feats_test_unscaled), # ### MODIFIED: .fit_transform ###
            index=interaction_all_feats_test_unscaled.index,
            columns=interaction_all_feats_test_unscaled.columns
        )

        print("\n--- DIAGNOSTIC SHAPES (After Scaling Test Data) ---")
        print(f"Shape of siRNA .x features scaled: {sirna_all_feats_scaled.shape}")
        print(f"Shape of mRNA .x features scaled: {mrna_all_feats_scaled.shape}")
        print(f"Shape of Interaction .x features scaled: {interaction_all_feats_scaled.shape}")
        print("--- END DIAGNOSTIC ---\n")

        # Step 3: Build HeteroData Graph
        data_hetero = HeteroData()

        data_hetero['siRNA'].x = torch.tensor(sirna_all_feats_scaled.values, dtype=torch.float)
        data_hetero['mRNA'].x = torch.tensor(mrna_all_feats_scaled.values, dtype=torch.float)
        data_hetero['interaction'].x = torch.tensor(interaction_all_feats_scaled.values, dtype=torch.float)

        data_hetero['siRNA'].tokens = torch.tensor(sirna_tokens_df_test.values, dtype=torch.long)
        data_hetero['mRNA'].tokens = torch.tensor(mrna_tokens_df_test.values, dtype=torch.long)

        siRNA_name_to_idx_test = {name: i for i, name in enumerate(sirna_all_feats_scaled.index)}
        mRNA_name_to_idx_test = {name: i for i, name in enumerate(mrna_all_feats_scaled.index)}
        interaction_name_to_idx_test = {name: i for i, name in enumerate(interaction_all_feats_scaled.index)}

        edge_src_si = [siRNA_name_to_idx_test[name] for name in data['siRNA']]
        edge_dst_i = [interaction_name_to_idx_test[name] for name in interaction_index_map_test]
        data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor([edge_src_si, edge_dst_i], dtype=torch.long).contiguous()
        
        edge_src_m = [mRNA_name_to_idx_test[name] for name in data['mRNA']]
        data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor([edge_src_m, edge_dst_i], dtype=torch.long).contiguous()
        
        data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
        data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

        efficacy_map_test = data.set_index(['siRNA', 'mRNA'])['efficacy'].to_dict()
        ordered_labels = []
        for interaction_key in interaction_all_feats_scaled.index:
            try:
                siRNA_id, mRNA_id = interaction_key.split('_', 1)
                ordered_labels.append(efficacy_map_test[(siRNA_id, mRNA_id)])
            except KeyError:
                print(f"CRITICAL WARNING: Efficacy for interaction {interaction_key} not found in map. This may cause label/feature mismatch!")
                pass
        assert len(ordered_labels) == interaction_all_feats_scaled.shape[0], \
            f"Label count mismatch! Expected {interaction_all_feats_scaled.shape[0]}, got {len(ordered_labels)}."
        data_hetero['interaction'].y = torch.tensor(ordered_labels, dtype=torch.float)

        print(f"HeteroData Graph created: {data_hetero}")
        print(f"Number of siRNA nodes: {data_hetero['siRNA'].num_nodes}")
        print(f"Number of mRNA nodes: {data_hetero['mRNA'].num_nodes}")
        print(f"Number of interaction nodes: {data_hetero['interaction'].num_nodes}")
        print(f"data_hetero['interaction'].y shape: {data_hetero['interaction'].y.shape}")

        test_loader = NeighborLoader(
            data_hetero,
            num_neighbors=params["hop_samples"],
            batch_size=params["batch_size"],
            input_nodes=('interaction', torch.arange(data_hetero['interaction'].num_nodes)),
            shuffle=False,
            subgraph_type='induced',
            num_workers=params["num_workers"],
            filter_per_worker=params["filter_per_worker"]
        )

        # Step 4: Initialize Model and Load Weights
        sirna_other_feature_dim = sirna_all_feats_scaled.shape[1]
        mrna_other_feature_dim = mrna_all_feats_scaled.shape[1]
        interaction_feature_dim = interaction_all_feats_scaled.shape[1]
        
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
            sirna_other_feature_dim=sirna_other_feature_dim,
            mrna_other_feature_dim=mrna_other_feature_dim,
            interaction_feature_dim=interaction_feature_dim,
            dropout_rate=params["dropout"]
        ).to(DEVICE)

        model_path = os.path.join(MODEL_DIR, f"best_model_fold_{n}.pt")
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Successfully loaded model weights from: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: Model file not found at {model_path}. Make sure training script saved it. {e}")
            continue
        except Exception as e:
            print(f"An error occurred loading model for fold {n}: {e}")
            print("This is likely due to a feature mismatch or other model loading issue. Check model architecture and saved state_dict.")
            continue
        
        # ### --- MODIFICATION: Define efficacy_threshold directly for AUC calculation --- ###
        # Since scalers are not loaded, we cannot load a threshold from them.
        # Use a reasonable default or a threshold determined from training data analysis.
        efficacy_threshold = 0.5 # You might want to set this based on your training data's label distribution

        # Step 5: Run Inference and Calculate Metrics
        print("\nRunning inference on test data...")
        pcc, spcc, mse, auc = evaluate_model_for_test(model, test_loader, efficacy_threshold)
        
        all_pcc.append(pcc)
        all_spcc.append(spcc)
        all_mse.append(mse)
        all_auc.append(auc)
        print(f"Fold {n} Test Results -> PCC: {pcc:.4f} | SPCC: {spcc:.4f} | MSE: {mse:.4f} | AUC: {auc:.4f}")

    # --- Final Aggregate Results Across Folds ---
    if all_pcc:
        print(f"\n{'='*40}\n--- Final Aggregate Test Results (Averaged over {len(all_pcc)} folds) ---\n{'='*40}")
        print(f"Average PCC:    {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
        print(f"Average SPCC:   {np.mean(all_spcc):.4f} ± {np.std(all_spcc):.4f}")
        print(f"Average MSE:    {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
        valid_aucs = [auc_val for auc_val in all_auc if not np.isnan(auc_val)]
        avg_auc = np.mean(valid_aucs) if valid_aucs else np.nan
        print(f"Average AUC:    {avg_auc:.4f} ± {np.std(valid_aucs):.4f}" if valid_aucs else "Average AUC: N/A")
    else:
        print("\nNo folds were successfully tested. Please check model paths and data files.")

if __name__ == "__main__":
    main_test()