import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, GraphConv
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr
import json
torch.autograd.set_detect_anomaly(True)

torch.cuda.manual_seed_all(42)
# Import the revised utils1.py
import BugUtils as utils1

# --- Load parameters ---
with open("siRNA_param_pytorch.json", 'r') as f:
    params = json.load(f)

# Define MAX_SIRNA_LENGTH from parameters for clarity in feature generation
MAX_SIRNA_LENGTH = params["sirna_length"]
MAX_MRNA_LENGTH = params["max_mrna_len"]

# ------------------------
# 1. Sequence Encoder
# ------------------------
class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return h  # shape: (batch_size, embed_dim)

class EfficacyModel(nn.Module):
    def __init__(self, input_dim_mRNA, input_dim_siRNA, input_dim_inter,
                 proj_dim=128, hidden_dim=64, embed_dim=32, fusion_dim=128, mlp_dim=64):
        super().__init__()
        
        # Projection layers to map into same space
        self.proj_mRNA = nn.Linear(input_dim_mRNA, proj_dim)
        self.proj_siRNA = nn.Linear(input_dim_siRNA, proj_dim)
        self.ln_proj = nn.LayerNorm(proj_dim)   # shared LayerNorm for projections
        
        # Shared Siamese encoder (weight sharing)
        self.shared_encoder = SequenceEncoder(proj_dim, hidden_dim, embed_dim)
        
        # Fusion layer (siRNA + mRNA embeddings)
        self.fc_fusion = nn.Linear(embed_dim * 2, fusion_dim)  
        self.ln_fusion = nn.LayerNorm(fusion_dim)

        # MLP head (fusion + thermo)
        self.fc_mlp1 = nn.Linear(fusion_dim + input_dim_inter, mlp_dim)
        self.ln_mlp1 = nn.LayerNorm(mlp_dim)
        self.fc_out = nn.Linear(mlp_dim, 1)

    def forward(self, mRNA_x, siRNA_x, inter_x):
        # Step 1: project inputs into same dimension + layer norm
        mRNA_proj = self.ln_proj(self.proj_mRNA(mRNA_x))
        siRNA_proj = self.ln_proj(self.proj_siRNA(siRNA_x))
        
        # Step 2: pass through shared encoder
        mRNA_hidden = self.shared_encoder(mRNA_proj)   # (B, embed_dim)
        siRNA_hidden = self.shared_encoder(siRNA_proj) # (B, embed_dim)
        
        # Step 3: fuse embeddings
        fusion = torch.cat([mRNA_hidden, siRNA_hidden], dim=-1)  # (B, embed_dim*2)
        h = self.ln_fusion(F.relu(self.fc_fusion(fusion)))       # (B, fusion_dim)

        # Step 4: add thermo features
        h_inter = torch.cat([h, inter_x], dim=-1)              # (B, fusion_dim + thermo_dim)

        # Step 5: pass through MLP before regression
        h_mlp = self.ln_mlp1(F.relu(self.fc_mlp1(h_inter)))     # (B, mlp_dim)
        out = self.fc_out(h_mlp)                                 # regression score

        return out.squeeze(-1), h


class EfficacyDataset(Dataset):
    def __init__(self, df, sirna_pd, mrna_pd, inter_pd):
        """
        df: DataFrame (train or dev split) with at least ['siRNA','mRNA','efficacy']
        sirna_pd: DataFrame of siRNA features, indexed by siRNA ID
        mrna_pd: DataFrame of mRNA features, indexed by mRNA ID
        thermo_pd: DataFrame of thermo features, indexed by 'siRNA_mRNA'
        """
        self.df = df.reset_index(drop=True)
        self.sirna_pd = sirna_pd
        self.mrna_pd = mrna_pd
        self.inter_pd = inter_pd

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sirna_id = row["siRNA"].strip()
        mrna_id = row["mRNA"].strip()
        pair_id = f"{sirna_id}_{mrna_id}"

        # Lookup features
        sirna_x = torch.tensor(self.sirna_pd.loc[sirna_id].values, dtype=torch.float32)
        mrna_x  = torch.tensor(self.mrna_pd.loc[mrna_id].values, dtype=torch.float32)

        if pair_id not in self.inter_pd.index:
            raise KeyError(f"Thermo features missing for {pair_id}")
        inter_x = torch.tensor(self.inter_pd.loc[pair_id].values, dtype=torch.float32)

        label = torch.tensor(row["efficacy"], dtype=torch.float32)
        return mrna_x, sirna_x, inter_x, label

for n in range(10):
    print(f"\n--- Testing Fold {n} ---")

    # Load the single CSV file for testing
    test_file = "siRNA_mRNA_all_data.csv" # <--- IMPORTANT: Adjust this path to your actual test data file
    try:
        data_test = pd.read_csv(test_file)
        print(f"Successfully loaded test data from: {test_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {test_file}. Skipping fold {n}.")
        continue
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}. Skipping fold {n}.")
        continue

    for df in [data_test]: # Only train and dev
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')

    print("\n--- Feature Processing ---")

    # 1. One-hot encoding for siRNA (PADDED to MAX_SIRNA_LENGTH)
    sirna_onehot = []
    for seq in data_test['siRNA_seq']:
        sirna_onehot.append(utils1.obtain_one_hot_feature_for_one_sequence_1(seq, MAX_SIRNA_LENGTH))
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data_test['siRNA']))
 

    # 2. mRNA one-hot (PADDED to max_mrna_len)
    mrna_onehot_temp = data_test.loc[:, ['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils1.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"])
                   for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))

    # 3. Positional encoding (PADDED to MAX_SIRNA_LENGTH * dmodel)
    sirna_pos_encoding = []
    for idx, row in data_test.iterrows():
        mrna_start_pos = max(0, int(row['pos']))
        sirna_pos_encoding.append(utils1.get_pos_embedding_sequence(
            mrna_start_pos,
            len(row['siRNA_seq']),
            MAX_SIRNA_LENGTH,
            params["dmodel"]
        ))
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=data_test['siRNA'] + '_' + data_test['mRNA'])


    # 4. Thermodynamics (PADDED to MAX_SIRNA_LENGTH)
    sirna_thermo_feat = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH)
                         for seq in data_test['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    temp_interaction_index = data_test['siRNA'] + '_' + data_test['mRNA']
    sirna_thermo_feat['index'] = temp_interaction_index
    sirna_thermo_feat = sirna_thermo_feat.set_index('index')


    # Co-fold features
    try:
        con_feat = pd.read_csv("Simone_split_preprocess/con_matrix_Simone_meanSum50.txt", header=None, index_col=0)
        con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)
    except FileNotFoundError:
        print("Warning: co-fold feature file 'Simone_split_preprocess/con_matrix_Simone_meanSum50.txt' not found. Filling with zeros.")
        con_feat = pd.DataFrame(0.0, index=sirna_thermo_feat.index, columns=[f'cofold_dim_{i}' for i in range(50)])
    
    # self-fold features
    ## siRNA
    try:
        sirna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt", header=None, index_col=0)
        sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index).fillna(0)
    except FileNotFoundError:
        print("Warning: siRNA self-fold feature file 'Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt' not found. Filling with zeros.")
        sirna_sfold_feat = pd.DataFrame(0.0, index=sirna_embedding_df.index, columns=[f'sfold_siRNA_dim_{i}' for i in range(6)])

    ## mRNA
    try:
        mrna_sfold_feat = pd.read_csv("Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt", header=None, index_col=0)
        mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0)
    except FileNotFoundError:
        print("Warning: mRNA self-fold feature file 'Simone_split_preprocess/self_mRNA_matrix_Simone_meanSum100.txt' not found. Filling with zeros.")
        mrna_sfold_feat = pd.DataFrame(0.0, index=mrna_embedding_df.index, columns=[f'sfold_mRNA_dim_{i}' for i in range(100)])

    # AGO2
    ## siRNA-AGO2
    sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2_zh.csv",index_col = 0)

    sirna_ago = sirna_ago.reindex(sirna_onehot.index)
    has_nan = sirna_ago.isnull().values.any()
    print(f"sirna_ago Contains NaN: {has_nan}")

    ## mRNA-AGO2
    mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2_zh.csv",index_col=0)

    mrna_ago = mrna_ago.reindex(mrna_onehot.index)
    has_nan = mrna_ago.isnull().values.any()
    print(f"mrna_ago Contains NaN: {has_nan}")


    # 8. GC percentage (variable length robust)
    sirna_GC = pd.DataFrame([utils1.countGC(seq) for seq in data_test['siRNA_seq']], index=list(data_test['siRNA']))
    mrna_GC = pd.DataFrame([utils1.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq_RNA-FM']], index=list(mrna_onehot_temp['mRNA']))

    # 9. K-mers (All now return fixed-size lists)
    sirna_1_mer = pd.DataFrame([utils1.single_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils1.double_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils1.triple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils1.quadruple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils1.quintuple_freq(seq, MAX_SIRNA_LENGTH) for seq in data_test['siRNA_seq']])
    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data_test['siRNA']

    

    # 10. siRNA rules codes (PADDED to 19*3)
    SIRNA_RULES_LENGTH = 19
    sirna_pos_scores = []
    for seq in data_test['siRNA_seq']:
        sirna_pos_scores.append(utils1.rules_scores(seq, SIRNA_RULES_LENGTH))
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data_test['siRNA']))
    # print(f"\n--- siRNA Rules Scores ---")
    # print(f"siRNA Sequence: {first_siRNA_seq}")
    # print(f"Full DataFrame Shape: {sirna_pos_scores.shape}")
    # print(f"First Sample's Feature Vector (all values): {sirna_pos_scores.iloc[0].values}")

    print("\n--- Assembling GNN Node Features ---")

    # siRNA nodes features
    sirna_pd = pd.concat([sirna_onehot, sirna_sfold_feat, sirna_ago, sirna_GC, sirna_k_mers, sirna_pos_scores], axis=1)

    # mRNA nodes features
    mrna_pd = pd.concat([mrna_onehot, mrna_sfold_feat, mrna_ago, mrna_GC], axis=1)

    interaction_pd = pd.concat([sirna_thermo_feat,sirna_pos_encoding],axis=1)

    input_dim_mRNA = mrna_pd.shape[1]
    # print(input_dim_mRNA)
    input_dim_siRNA = sirna_pd.shape[1]

    input_dim_inter = interaction_pd.shape[1]
    

    def collate_fn(batch):
        mrna_list, sirna_list, inter_list, labels = zip(*batch)

        # Pad sequences
        mrna_padded = pad_sequence(mrna_list, batch_first=True)    # (B, Lm_max, d_mRNA)
        sirna_padded = pad_sequence(sirna_list, batch_first=True)  # (B, Ls_max, d_siRNA)
        inter_padded = pad_sequence(inter_list, batch_first=True) # (B, Lt_max, d_thermo)

        # Masks
        mrna_mask = torch.zeros(mrna_padded.shape[:2], dtype=torch.bool)
        sirna_mask = torch.zeros(sirna_padded.shape[:2], dtype=torch.bool)
        inter_mask = torch.zeros(inter_padded.shape[:2], dtype=torch.bool)

        for i, (m, s, t) in enumerate(zip(mrna_list, sirna_list, inter_list)):
            mrna_mask[i, :len(m)] = 1
            sirna_mask[i, :len(s)] = 1
            inter_mask[i, :len(t)] = 1

        labels = torch.stack(labels)  # (B,)

        return mrna_padded, sirna_padded, inter_padded, labels, mrna_mask, sirna_mask, inter_mask

    def contrastive_loss(emb_m, emb_s, label, margin=1.0):
        dist = torch.norm(emb_m - emb_s, p=2, dim=1)
        if torch.isnan(dist).any():
            print("NaN detected in distance.")
        relu_term = F.relu(margin - dist)
        if torch.isnan(relu_term).any():
            print("NaN detected in relu term.")
        loss = label * dist.pow(2) + (1 - label) * relu_term.pow(2)
        if torch.isnan(loss).any():
            print("NaN detected in loss.")
        return loss.mean()

    test_dataset   = EfficacyDataset(data_test, sirna_pd, mrna_pd, interaction_pd)

    test_loader   = DataLoader(test_dataset, params["batch_size"], shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = EfficacyModel(input_dim_mRNA, input_dim_siRNA, input_dim_inter).cuda()

    # Load the saved weights
    model_path = f"best_model_fold0.pt" # <--- IMPORTANT: Ensure this path is correct for your saved models
    # model_path = f"best_model_fold1_gcn_75.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot test this fold.")
        continue # Skip to next fold
    except Exception as e:
        print(f"An error occurred while loading model for fold {n}: {e}. Skipping this fold.")
        continue

    model.eval()


    def validate(model, dataloader, device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                mRNA_x, siRNA_x, inter_x, y_true, mrna_mask, sirna_mask, inter_mask = [b.to(device) for b in batch]

                # Forward pass (with thermo_x)
                y_pred, _ = model(mRNA_x, siRNA_x, inter_x)

                all_preds.append(y_pred.cpu())
                all_labels.append(y_true.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Metrics calculation
        mse = mean_squared_error(all_labels, all_preds)
        pcc, _ = pearsonr(all_labels, all_preds)
        spcc, _ = spearmanr(all_labels, all_preds)
    
        # For classification metrics, binarize with threshold 0.7
        binary_true = (all_labels > 0.7).astype(int)
        binary_pred = (all_preds > 0.7).astype(int)
    
        try:
            auc = roc_auc_score(binary_true, all_preds)
        except ValueError:
            auc = float('nan')  # If roc_auc_score can't be computed
    
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)

        print(f"Validation MSE: {mse:.4f}")
        print(f"Validation PCC: {pcc:.4f}")
        print(f"Validation SPCC: {spcc:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")

        return mse, pcc, spcc, auc, f1, precision, recall

    mse,_,_,auc,_,_,_ = validate(model, test_loader, device="cuda")