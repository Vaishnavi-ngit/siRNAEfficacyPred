import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
import json
import math
from torch.cuda.amp import GradScaler, autocast
import scipy.stats
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import RobustScaler
import Utils as utils1 
from Notting import tokenize_sequence, TransformerEncoder, HeteroGNN, PAD_TOKEN_ID, CLS_TOKEN_ID


def main():
    print("Starting siRNA Efficacy Prediction Model (Integrated Pipeline)")
    try:
        with open("param.json", 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print("param.json not found, using default parameters.")
        params = {}
    MAX_SIRNA_LENGTH = params.setdefault("sirna_length", 31)
    MAX_MRNA_LENGTH = params.setdefault("max_mrna_len", 2001)

    # params.setdefault("embedding_dim", 256)
    # params.setdefault("dmodel", params["embedding_dim"])
    # params.setdefault("n_head", 8)
    # params.setdefault("n_layers", 4)
    # params.setdefault("gat_heads", 8)
    # params.setdefault("hinsage_layer_sizes", [256, 128, 64])
    # params.setdefault("dropout", 0.2)
    # params.setdefault("batch_size", 128)
    # params.setdefault("hop_samples", [20, 10])
    # params.setdefault("clip_grad_norm", 1.0)
    # params.setdefault("lr", 0.0005)
    # params.setdefault("weight_decay", 0.00001)
    # params.setdefault("epochs", 500)
    # params.setdefault("early_stopping_patience", 50)
    # params.setdefault("loss", "mse")
    # params.setdefault("l1_lambda", 0.0)

    # --- Test Data Paths (Simone Dataset) ---
    TEST_DATA_CSV_PATH = "../encoder_Simone/Simone_all_data.csv"
    TEST_SIRNA_SELF_MATRIX_PATH = "../encoder_Simone/Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt"
    TEST_MRNA_SELF_MATRIX_PATH = "../encoder_Simone/Simone_split_preprocess/self_mRNA_matrix.txt"
    TEST_CON_MATRIX_PATH = "../encoder_Simone/Simone_split_preprocess/con_matrix.txt"

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    print(f"Parameters: {params}")

    # --- GLOBAL TEST DATA LOADING & PRE-PROCESSING (ONE-TIME) ---
    print("\n--- Pre-loading Test Dataset (Simone) ---")
    data_test_raw_global = None # Initialize to None
    try:
        data_test_raw_global = pd.read_csv(TEST_DATA_CSV_PATH)
        data_test_raw_global['siRNA_seq'] = data_test_raw_global['siRNA_seq'].str.replace('U', 'T').str.upper()
        data_test_raw_global['mRNA_seq_RNA-FM'] = data_test_raw_global['mRNA_seq_RNA-FM'].str.upper()
    except FileNotFoundError as e:
        print(f"Error: Global test data file not found at '{TEST_DATA_CSV_PATH}'. Skipping test set evaluation entirely. {e}")
    
    # --- K-Fold Cross-Validation Setup ---
    NUM_FOLDS = 10
    all_fold_results = []       # Stores validation results per fold (e.g., best val loss)
    all_test_results = []       # Stores test results per fold (best test metrics matching best val epoch)

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

        for df in [data_train_raw, data_dev_raw]:
            df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T').str.upper()
            df['mRNA_seq_RNA-FM'] = df['mRNA_seq_RNA-FM'].str.upper()

        data_combined = pd.concat([data_train_raw, data_dev_raw], axis=0).reset_index(drop=True)

        print("\n--- Feature Processing ---")
        unique_sirna_ids = data_combined['siRNA'].drop_duplicates().reset_index(drop=True)
        unique_mrna_ids = data_combined['mRNA'].drop_duplicates().reset_index(drop=True)
        unique_sirna_seq_map = data_combined[['siRNA', 'siRNA_seq']].drop_duplicates(subset='siRNA').set_index('siRNA')
        unique_mrna_seq_map = data_combined[['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset='mRNA').set_index('mRNA')

        # --- Features for TransformerEncoder (GC + K-mers) ---
        sirna_gc_kmers = []
        for seq_id in unique_sirna_seq_map.index:
            seq = unique_sirna_seq_map.loc[seq_id, 'siRNA_seq']
            gc = utils1.countGC(seq)
            kmer_freqs = utils1.get_kmer_freq(seq, k=1)
            kmer_freqs.extend(utils1.get_kmer_freq(seq, k=2))
            sirna_gc_kmers.append([gc] + kmer_freqs)
        sirna_gc_kmers_df = pd.DataFrame(sirna_gc_kmers, index=unique_sirna_seq_map.index)

        mrna_gc_kmers = []
        for seq_id in unique_mrna_seq_map.index:
            seq = unique_mrna_seq_map.loc[seq_id, 'mRNA_seq_RNA-FM']
            gc = utils1.countGC(seq)
            kmer_freqs = utils1.get_kmer_freq(seq, k=1)
            kmer_freqs.extend(utils1.get_kmer_freq(seq, k=2))
            mrna_gc_kmers.append([gc] + kmer_freqs)
        mrna_gc_kmers_df = pd.DataFrame(mrna_gc_kmers, index=unique_mrna_seq_map.index)

        # --- Other Node Features (siRNA.x, mRNA.x, interaction.x) ---
        # FIXED: Removed try-except for these critical feature files - if they're missing, it's an error.
        sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_siRNA_matrix.txt", header=None, index_col=0)
        sirna_sfold_feat = sirna_sfold_feat.reindex(unique_sirna_seq_map.index).fillna(0)
        sirna_pos_scores = pd.DataFrame([utils1.rules_scores(seq, params["sirna_length"]) for seq in unique_sirna_seq_map['siRNA_seq']],
                                         index=unique_sirna_seq_map.index)
        sirna_other_feats = pd.concat([sirna_sfold_feat, sirna_pos_scores], axis=1).fillna(0)

        mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/full_self_mRNA_matrix.txt", header=None, index_col=0)
        mrna_sfold_feat = mrna_sfold_feat.reindex(unique_mrna_seq_map.index).fillna(0)
        mrna_rnafm_embeddings = pd.DataFrame([utils1.get_mp_rna_sequence_embedding(seq) for seq in unique_mrna_seq_map['mRNA_seq_RNA-FM']],
                                             index=unique_mrna_seq_map.index)
        mrna_other_feats = pd.concat([mrna_sfold_feat, mrna_rnafm_embeddings], axis=1).fillna(0)

        # --- Interaction features ---
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
                mrna_start_pos, len(row['siRNA_seq']), MAX_SIRNA_LENGTH, params["dmodel"]
            ))
        sirna_pos_encoding_df = pd.DataFrame(sirna_pos_encoding).set_index(interaction_index_map)
        interaction_all_feats = pd.concat([sirna_thermo_feat_df, con_feat, sirna_pos_encoding_df], axis=1).fillna(0)

        # --- Feature Scaling (RobustScaler) ---
        print("\nScaling features for current fold...")
        scaler_siRNA = RobustScaler()
        scaler_mRNA = RobustScaler()
        scaler_interaction = RobustScaler()
        scaler_siRNA_gc_kmer = RobustScaler()
        scaler_mRNA_gc_kmer = RobustScaler()
        scaler_efficacy = RobustScaler() # Scaler for the target variable

        sirna_other_feats_scaled = pd.DataFrame(scaler_siRNA.fit_transform(sirna_other_feats),
                                                index=sirna_other_feats.index, columns=sirna_other_feats.columns)
        mrna_other_feats_scaled = pd.DataFrame(scaler_mRNA.fit_transform(mrna_other_feats),
                                               index=mrna_other_feats.index, columns=mrna_other_feats.columns)
        interaction_all_feats_scaled = pd.DataFrame(scaler_interaction.fit_transform(interaction_all_feats),
                                                    index=interaction_all_feats.index, columns=interaction_all_feats.columns)
        sirna_gc_kmers_scaled = pd.DataFrame(scaler_siRNA_gc_kmer.fit_transform(sirna_gc_kmers_df),
                                            index=sirna_gc_kmers_df.index, columns=sirna_gc_kmers_df.columns)
        mrna_gc_kmers_scaled = pd.DataFrame(scaler_mRNA_gc_kmer.fit_transform(mrna_gc_kmers_df),
                                            index=mrna_gc_kmers_df.index, columns=mrna_gc_kmers_df.columns)
        
        # --- FIXED: Efficacy Scaling ---
        efficacy_combined_fold_df = data_combined[['efficacy']] 
        efficacy_scaled_values = scaler_efficacy.fit_transform(efficacy_combined_fold_df)
        scaled_efficacy_series = pd.Series(efficacy_scaled_values.flatten(), index=data_combined.index)


        # --- Construct HeteroData for this fold's TRAIN/DEV ---
        print("\n--- Constructing HeteroData Graph for Train/Dev ---")
        data_hetero = HeteroData()
        data_hetero['siRNA'].x = torch.tensor(sirna_other_feats_scaled.reindex(unique_sirna_ids).values, dtype=torch.float)
        data_hetero['mRNA'].x = torch.tensor(mrna_other_feats_scaled.reindex(unique_mrna_ids).values, dtype=torch.float)
        data_hetero['interaction'].x = torch.tensor(interaction_all_feats_scaled.values, dtype=torch.float)

        data_hetero['siRNA'].tokens = torch.tensor([tokenize_sequence(seq, MAX_SIRNA_LENGTH) for seq in unique_sirna_seq_map['siRNA_seq']], dtype=torch.long)
        data_hetero['mRNA'].tokens = torch.tensor([tokenize_sequence(seq, MAX_MRNA_LENGTH) for seq in unique_mrna_seq_map['mRNA_seq_RNA-FM']], dtype=torch.long)
        data_hetero['siRNA'].seq_features = torch.tensor(sirna_gc_kmers_scaled.reindex(unique_sirna_ids).values, dtype=torch.float)
        data_hetero['mRNA'].seq_features = torch.tensor(mrna_gc_kmers_scaled.reindex(unique_mrna_ids).values, dtype=torch.float)

        siRNA_name_to_idx = {name: i for i, name in enumerate(unique_sirna_ids)}
        mRNA_name_to_idx = {name: i for i, name in enumerate(unique_mrna_ids)}
        interaction_name_to_idx = {name: i for i, name in enumerate(interaction_all_feats_scaled.index)}

        edge_index_siRNA_to_interaction = []
        edge_index_mRNA_to_interaction = []
        for _, row in data_combined.iterrows():
            siRNA_name = row['siRNA']
            mRNA_name = row['mRNA']
            interaction_name = f"{siRNA_name}_{mRNA_name}"
            if (siRNA_name in siRNA_name_to_idx and mRNA_name in mRNA_name_to_idx and interaction_name in interaction_name_to_idx):
                edge_index_siRNA_to_interaction.append([siRNA_name_to_idx[siRNA_name], interaction_name_to_idx[interaction_name]])
                edge_index_mRNA_to_interaction.append([mRNA_name_to_idx[mRNA_name], interaction_name_to_idx[interaction_name]])
            else:
                print(f"Warning: Skipping edge for {interaction_name} due to missing node in feature dataframes or index mismatch.")

        data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
        data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
        data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
        data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

        # --- FIXED: Efficacy label assignment using the scaled series ---
        final_ordered_scaled_labels = []
        for interaction_key in interaction_all_feats_scaled.index:
            try:
                original_row_idx = data_combined.index[data_combined.apply(lambda row: f"{row['siRNA']}_{row['mRNA']}", axis=1) == interaction_key].item()
                final_ordered_scaled_labels.append(scaled_efficacy_series.loc[original_row_idx])
            except (KeyError, ValueError): 
                print(f"CRITICAL WARNING: Scaled efficacy for interaction {interaction_key} not found for label assignment. This is an inconsistency.")
                pass 
        assert len(final_ordered_scaled_labels) == interaction_all_feats_scaled.shape[0], f"Label count mismatch! Expected {interaction_all_feats_scaled.shape[0]}, got {len(final_ordered_scaled_labels)}."
        data_hetero['interaction'].y = torch.tensor(final_ordered_scaled_labels, dtype=torch.float)

        train_interaction_indices = [interaction_name_to_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train_raw.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_name_to_idx]
        dev_interaction_indices = [interaction_name_to_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev_raw.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_name_to_idx]
        train_idx = torch.tensor(train_interaction_indices, dtype=torch.long)
        dev_idx = torch.tensor(dev_interaction_indices, dtype=torch.long)

        print(f"HeteroData Graph created: {data_hetero}")
        print(f"Number of siRNA nodes: {data_hetero['siRNA'].num_nodes}")
        print(f"Number of mRNA nodes: {data_hetero['mRNA'].num_nodes}")
        print(f"Number of interaction nodes: {data_hetero['interaction'].num_nodes}")
        print(f"Number of training interactions: {len(train_idx)}")
        print(f"Number of validation interactions: {len(dev_idx)}")

        train_loader = NeighborLoader(data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"],
                                    input_nodes=('interaction', train_idx), shuffle=True, subgraph_type='induced', num_workers=0)
        val_loader = NeighborLoader(data_hetero, num_neighbors=params["hop_samples"], batch_size=params["batch_size"],
                                  input_nodes=('interaction', dev_idx), shuffle=False, subgraph_type='induced', num_workers=0)

        # --- Construct HeteroData & Loader for TEST set (once per fold, using current scalers) ---
        test_loader = None
        if data_test_raw_global is not None:
            print("\n--- Constructing HeteroData Graph for Test Set (using current fold's scalers) ---")

            unique_sirna_ids_test = data_test_raw_global['siRNA'].drop_duplicates().reset_index(drop=True)
            unique_mrna_ids_test = data_test_raw_global['mRNA'].drop_duplicates().reset_index(drop=True)
            unique_sirna_seq_map_test = data_test_raw_global[['siRNA', 'siRNA_seq']].drop_duplicates(subset='siRNA').set_index('siRNA')
            unique_mrna_seq_map_test = data_test_raw_global[['mRNA', 'mRNA_seq_RNA-FM']].drop_duplicates(subset='mRNA').set_index('mRNA')

            # --- Generate Test Features (CRUCIAL: USE TRAINED SCALERS' TRANSFORM) ---
            sirna_gc_kmers_test = []
            for seq_id in unique_sirna_seq_map_test.index:
                seq = unique_sirna_seq_map_test.loc[seq_id, 'siRNA_seq']
                gc = utils1.countGC(seq)
                kmer_freqs = utils1.get_kmer_freq(seq, k=1)
                kmer_freqs.extend(utils1.get_kmer_freq(seq, k=2))
                sirna_gc_kmers_test.append([gc] + kmer_freqs)
            sirna_gc_kmers_df_test = pd.DataFrame(sirna_gc_kmers_test, index=unique_sirna_seq_map_test.index)
            sirna_gc_kmers_scaled_test = pd.DataFrame(scaler_siRNA_gc_kmer.transform(sirna_gc_kmers_df_test),
                                                      index=sirna_gc_kmers_df_test.index, columns=sirna_gc_kmers_df_test.columns)

            mrna_gc_kmers_test = []
            for seq_id in unique_mrna_seq_map_test.index:
                seq = unique_mrna_seq_map_test.loc[seq_id, 'mRNA_seq_RNA-FM']
                gc = utils1.countGC(seq)
                kmer_freqs = utils1.get_kmer_freq(seq, k=1)
                kmer_freqs.extend(utils1.get_kmer_freq(seq, k=2))
                mrna_gc_kmers_test.append([gc] + kmer_freqs)
            mrna_gc_kmers_df_test = pd.DataFrame(mrna_gc_kmers_test, index=unique_mrna_seq_map_test.index)
            mrna_gc_kmers_scaled_test = pd.DataFrame(scaler_mRNA_gc_kmer.transform(mrna_gc_kmers_df_test),
                                                      index=mrna_gc_kmers_df_test.index, columns=mrna_gc_kmers_df_test.columns)

            # Other siRNA features for test
            # FIXED: Removed try-except, assumes file exists. Added fallback for dummy shape to be consistent with train-side in case of missing original features
            sirna_sfold_feat_test = pd.read_csv(TEST_SIRNA_SELF_MATRIX_PATH, header=None, index_col=0)
            sirna_sfold_feat_test = sirna_sfold_feat_test.reindex(unique_sirna_seq_map_test.index).fillna(0)
            sirna_pos_scores_test = pd.DataFrame([utils1.rules_scores(seq, params["sirna_length"]) for seq in unique_sirna_seq_map_test['siRNA_seq']],
                                                 index=unique_sirna_seq_map_test.index)
            sirna_other_feats_test = pd.concat([sirna_sfold_feat_test, sirna_pos_scores_test], axis=1).fillna(0)
            sirna_other_feats_scaled_test = pd.DataFrame(scaler_siRNA.transform(sirna_other_feats_test),
                                                        index=sirna_other_feats_test.index, columns=sirna_other_feats_test.columns)

            # Other mRNA features for test
            # FIXED: Removed try-except
            mrna_sfold_feat_test = pd.read_csv(TEST_MRNA_SELF_MATRIX_PATH, header=None, index_col=0)
            mrna_sfold_feat_test = mrna_sfold_feat_test.reindex(unique_mrna_seq_map_test.index).fillna(0)
            mrna_rnafm_embeddings_test = pd.DataFrame([utils1.get_mp_rna_sequence_embedding(seq) for seq in unique_mrna_seq_map_test['mRNA_seq_RNA-FM']],
                                                      index=unique_mrna_seq_map_test.index)
            mrna_other_feats_test = pd.concat([mrna_sfold_feat_test, mrna_rnafm_embeddings_test], axis=1).fillna(0)
            mrna_other_feats_scaled_test = pd.DataFrame(scaler_mRNA.transform(mrna_other_feats_test),
                                                        index=mrna_other_feats_test.index, columns=mrna_other_feats_test.columns)

            # Interaction features for test
            interaction_index_map_test = data_test_raw_global.apply(lambda row: f"{row['siRNA']}_{row['mRNA']}", axis=1)
            sirna_thermo_feat_list_test = [utils1.cal_thermo_feature(seq, MAX_SIRNA_LENGTH) for seq in data_test_raw_global['siRNA_seq']]
            sirna_thermo_feat_df_test = pd.DataFrame(sirna_thermo_feat_list_test).set_index(interaction_index_map_test)
            # FIXED: Removed try-except
            con_feat_test = pd.read_csv(TEST_CON_MATRIX_PATH, header=None, index_col=0)
            con_feat_test = con_feat_test.reindex(interaction_index_map_test).fillna(0)
            sirna_pos_encoding_test = []
            for idx, row in data_test_raw_global.iterrows():
                mrna_start_pos = max(0, int(row['pos']))
                sirna_pos_encoding_test.append(utils1.get_pos_embedding_sequence(
                    mrna_start_pos, len(row['siRNA_seq']), MAX_SIRNA_LENGTH, params["dmodel"]))
            sirna_pos_encoding_df_test = pd.DataFrame(sirna_pos_encoding_test).set_index(interaction_index_map_test)
            interaction_all_feats_test = pd.concat([sirna_thermo_feat_df_test, con_feat_test, sirna_pos_encoding_df_test], axis=1).fillna(0)
            interaction_all_feats_scaled_test = pd.DataFrame(scaler_interaction.transform(interaction_all_feats_test),
                                                            index=interaction_all_feats_test.index, columns=interaction_all_feats_test.columns)

            # --- Construct HeteroData for Test Set ---
            data_hetero_test = HeteroData()
            data_hetero_test['siRNA'].x = torch.tensor(sirna_other_feats_scaled_test.reindex(unique_sirna_ids_test).values, dtype=torch.float)
            data_hetero_test['mRNA'].x = torch.tensor(mrna_other_feats_scaled_test.reindex(unique_mrna_ids_test).values, dtype=torch.float)
            data_hetero_test['interaction'].x = torch.tensor(interaction_all_feats_scaled_test.values, dtype=torch.float)
            data_hetero_test['siRNA'].tokens = torch.tensor([tokenize_sequence(seq, MAX_SIRNA_LENGTH) for seq in unique_sirna_seq_map_test['siRNA_seq']], dtype=torch.long)
            data_hetero_test['mRNA'].tokens = torch.tensor([tokenize_sequence(seq, MAX_MRNA_LENGTH) for seq in unique_mrna_seq_map_test['mRNA_seq_RNA-FM']], dtype=torch.long)
            data_hetero_test['siRNA'].seq_features = torch.tensor(sirna_gc_kmers_scaled_test.reindex(unique_sirna_ids_test).values, dtype=torch.float)
            data_hetero_test['mRNA'].seq_features = torch.tensor(mrna_gc_kmers_scaled_test.reindex(unique_mrna_ids_test).values, dtype=torch.float)

            siRNA_name_to_idx_test = {name: i for i, name in enumerate(unique_sirna_ids_test)}
            mRNA_name_to_idx_test = {name: i for i, name in enumerate(unique_mrna_ids_test)}
            interaction_name_to_idx_test = {name: i for i, name in enumerate(interaction_all_feats_scaled_test.index)}

            edge_index_siRNA_to_interaction_test = []
            edge_index_mRNA_to_interaction_test = []
            for _, row in data_test_raw_global.iterrows():
                siRNA_name = row['siRNA']
                mRNA_name = row['mRNA']
                interaction_name = f"{siRNA_name}_{mRNA_name}"
                if (siRNA_name in siRNA_name_to_idx_test and mRNA_name in mRNA_name_to_idx_test and interaction_name in interaction_name_to_idx_test):
                    edge_index_siRNA_to_interaction_test.append([siRNA_name_to_idx_test[siRNA_name], interaction_name_to_idx_test[interaction_name]])
                    edge_index_mRNA_to_interaction_test.append([mRNA_name_to_idx_test[mRNA_name], interaction_name_to_idx_test[interaction_name]])
                else:
                    print(f"Warning: Skipping test edge for {interaction_name} due to missing node in feature dataframes or index mismatch.")
            data_hetero_test['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(edge_index_siRNA_to_interaction_test, dtype=torch.long).t().contiguous()
            data_hetero_test['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(edge_index_mRNA_to_interaction_test, dtype=torch.long).t().contiguous()
            data_hetero_test['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero_test['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
            data_hetero_test['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero_test['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

            # --- FIXED: Efficacy label assignment using the scaled test efficacy ---
            efficacy_test_raw_df = data_test_raw_global[['efficacy']]
            efficacy_scaled_test_values = scaler_efficacy.transform(efficacy_test_raw_df)
            scaled_efficacy_series_test = pd.Series(efficacy_scaled_test_values.flatten(), index=data_test_raw_global.index)
            
            final_ordered_scaled_labels_test = []
            for interaction_key in interaction_all_feats_scaled_test.index:
                try:
                    original_row_idx_test = data_test_raw_global.index[data_test_raw_global.apply(lambda row: f"{row['siRNA']}_{row['mRNA']}", axis=1) == interaction_key].item()
                    final_ordered_scaled_labels_test.append(scaled_efficacy_series_test.loc[original_row_idx_test])
                except (KeyError, ValueError):
                    print(f"CRITICAL WARNING: Scaled efficacy for test interaction {interaction_key} not found for label assignment. This is an inconsistency.")
                    pass
            assert len(final_ordered_scaled_labels_test) == interaction_all_feats_scaled_test.shape[0], f"Test label count mismatch! Expected {interaction_all_feats_scaled_test.shape[0]}, got {len(final_ordered_scaled_labels_test)}."
            data_hetero_test['interaction'].y = torch.tensor(final_ordered_scaled_labels_test, dtype=torch.float)

            test_loader = NeighborLoader(data_hetero_test, num_neighbors=params["hop_samples"], batch_size=params["batch_size"],
                                        input_nodes=('interaction', None), shuffle=False, subgraph_type='induced', num_workers=0)
        else:
            test_loader = None
            print("Test data not available. Skipping per-epoch test evaluation.")

        # --- Model Instantiation and Training Setup ---
        sirna_other_feature_dim = data_hetero['siRNA'].x.shape[1]
        mrna_other_feature_dim = data_hetero['mRNA'].x.shape[1]
        interaction_feature_dim = data_hetero['interaction'].x.shape[1]
        sirna_seq_feat_dim = data_hetero['siRNA'].seq_features.shape[1]
        mrna_seq_feat_dim = data_hetero['mRNA'].seq_features.shape[1]

        model = HeteroGNN(
            vocab_size=6,
            embedding_dim=params["embedding_dim"], n_head=params["n_head"], n_layers=params["n_layers"],
            gat_heads=params["gat_heads"], layer_sizes=params["hinsage_layer_sizes"], out_channels=1,
            metadata=data_hetero.metadata(),
            sirna_seq_len=MAX_SIRNA_LENGTH, mrna_seq_len=MAX_MRNA_LENGTH,
            sirna_other_feature_dim=sirna_other_feature_dim, mrna_other_feature_dim=mrna_other_feature_dim,
            sirna_seq_feat_dim=sirna_seq_feat_dim, mrna_seq_feat_dim=mrna_seq_feat_dim,
            interaction_feature_dim=interaction_feature_dim, dropout_rate=params["dropout"]
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
        criterion = nn.MSELoss() if params["loss"] == "mse" else nn.SmoothL1Loss()
        scaler = GradScaler()

        best_val_loss_fold = float('inf')
        best_epoch_test_results = {}
        patience_counter = 0

        for epoch in range(params["epochs"]):
            # Training Phase
            train_loss = 0
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    token_dict = {'siRNA': batch['siRNA'].tokens, 'mRNA': batch['mRNA'].tokens}
                    seq_feat_dict = {'siRNA': batch['siRNA'].seq_features, 'mRNA': batch['mRNA'].seq_features}
                    out = model(batch.x_dict, batch.edge_index_dict, token_dict, seq_feat_dict).squeeze()
                    loss = criterion(out, batch['interaction'].y) 

                    l1_lambda = params.get("l1_lambda", 0.0)
                    if l1_lambda > 0:
                        l1_reg = torch.tensor(0.0, device=DEVICE)
                        for param in model.parameters(): l1_reg += torch.norm(param, 1)
                        loss += l1_lambda * l1_reg

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation Phase
            val_loss = 0 
            val_preds_list, val_targets_list = [], []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    token_dict = {'siRNA': batch['siRNA'].tokens, 'mRNA': batch['mRNA'].tokens}
                    seq_feat_dict = {'siRNA': batch['siRNA'].seq_features, 'mRNA': batch['mRNA'].seq_features}
                    out = model(batch.x_dict, batch.edge_index_dict, token_dict, seq_feat_dict).squeeze()
                    loss = criterion(out, batch['interaction'].y)
                    val_loss += loss.item()
                    val_preds_list.extend(out.cpu().numpy())
                    val_targets_list.extend(batch['interaction'].y.cpu().numpy())
                val_loss /= len(val_loader)

            # --- FIXED: Inverse Transform Validation Predictions for Metrics ---
            val_preds_scaled = np.array(val_preds_list)
            val_targets_scaled = np.array(val_targets_list)
            
            val_preds_original_scale = scaler_efficacy.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
            val_targets_original_scale = scaler_efficacy.inverse_transform(val_targets_scaled.reshape(-1, 1)).flatten()

            val_pcc = scipy.stats.pearsonr(val_targets_original_scale, val_preds_original_scale)[0]
            efficacy_threshold_val = np.median(val_targets_original_scale) 
            bin_val_targets = (val_targets_original_scale >= efficacy_threshold_val).astype(int)
            val_auc = roc_auc_score(bin_val_targets, val_preds_original_scale) if len(np.unique(bin_val_targets)) > 1 else np.nan
            val_rmse = np.sqrt(mean_squared_error(val_targets_original_scale, val_preds_original_scale))

            scheduler.step()

            # --- Test Phase (Per Epoch) ---
            current_test_metrics = {}
            if test_loader is not None:
                test_preds_list, test_targets_list = [], []
                test_loss_scaled = 0 
                model.eval() 
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(DEVICE)
                        token_dict_test = {'siRNA': batch['siRNA'].tokens, 'mRNA': batch['mRNA'].tokens}
                        seq_feat_dict_test = {'siRNA': batch['siRNA'].seq_features, 'mRNA': batch['mRNA'].seq_features}
                        out_test = model(batch.x_dict, batch.edge_index_dict, token_dict_test, seq_feat_dict_test).squeeze()
                        loss_test = criterion(out_test, batch['interaction'].y) 
                        test_loss_scaled += loss_test.item()
                        test_preds_list.extend(out_test.cpu().numpy())
                        test_targets_list.extend(batch['interaction'].y.cpu().numpy())
                    test_loss_scaled /= len(test_loader)

                # --- FIXED: Inverse Transform Test Predictions for Metrics ---
                test_preds_scaled = np.array(test_preds_list)
                test_targets_scaled = np.array(test_targets_list)
                
                test_preds_original_scale = scaler_efficacy.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
                test_targets_original_scale = scaler_efficacy.inverse_transform(test_targets_scaled.reshape(-1, 1)).flatten()

                test_rmse = np.sqrt(mean_squared_error(test_targets_original_scale, test_preds_original_scale))
                test_pcc = scipy.stats.pearsonr(test_targets_original_scale, test_preds_original_scale)[0]
                efficacy_threshold_test = np.median(test_targets_original_scale) 
                bin_test_targets = (test_targets_original_scale >= efficacy_threshold_test).astype(int)
                test_auc = roc_auc_score(bin_test_targets, test_preds_original_scale) if len(np.unique(bin_test_targets)) > 1 else np.nan

                current_test_metrics = {
                    'loss_scaled': test_loss_scaled, 
                    'rmse': test_rmse, 'pcc': test_pcc, 'auc': test_auc
                }

            # --- Early Stopping Logic ---
            if val_loss < best_val_loss_fold: 
                best_val_loss_fold = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold{n}.pth')
                if current_test_metrics: 
                    best_epoch_test_results = current_test_metrics.copy() 
            else:
                patience_counter += 1
                if patience_counter >= params["early_stopping_patience"]:
                    print(f"Early stopping triggered for fold {n+1} after {epoch + 1} epochs.")
                    break

            # Print progress every 10 epochs or at the last epoch, including test metrics
            log_string = (f"Fold {n+1}, Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | "
                          f"Val PCC: {val_pcc:.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            if current_test_metrics:
                log_string += (f" | Test Loss (Scaled): {current_test_metrics['loss_scaled']:.4f} | "
                               f"Test RMSE: {current_test_metrics['rmse']:.4f} | "
                               f"Test PCC: {current_test_metrics['pcc']:.4f} | "
                               f"Test AUC: {current_test_metrics['auc']:.4f}")
            if epoch % 10 == 0 or epoch == params["epochs"] - 1:
                print(log_string)

        # After epoch loop for the current fold, store best validation and corresponding test results
        all_fold_results.append({
            'fold': n, 'val_loss': best_val_loss_fold, 
        })
        if best_epoch_test_results:
            all_test_results.append({
                'fold': n,
                'test_loss_scaled': best_epoch_test_results.get('loss_scaled', np.nan),
                'test_rmse': best_epoch_test_results['rmse'],
                'test_correlation': best_epoch_test_results['pcc'],
                'test_auc': best_epoch_test_results['auc']
            })

    # --- Final Aggregate Results Summary ---
    print("\n--- Training and Evaluation Complete ---")

    avg_val_loss = np.mean([r['val_loss'] for r in all_fold_results])
    print(f"\nAverage Validation Loss (Scaled) across {NUM_FOLDS} folds: {avg_val_loss:.4f}")

    if all_test_results:
        avg_test_rmse = np.mean([r['test_rmse'] for r in all_test_results if 'test_rmse' in r and not np.isnan(r['test_rmse'])])
        avg_test_pcc = np.mean([r['test_correlation'] for r in all_test_results if 'test_correlation' in r and not np.isnan(r['test_correlation'])])
        valid_test_aucs = [r['test_auc'] for r in all_test_results if 'test_auc' in r and not np.isnan(r['test_auc'])]
        avg_test_auc = np.mean(valid_test_aucs) if valid_test_aucs else np.nan

        print("\n--- Final Average Test Results Across All Folds (from best validation epoch) ---")
        print(f"Average Test RMSE: {avg_test_rmse:.4f}")
        print(f"Average Test PCC: {avg_test_pcc:.4f}")
        print(f"Average Test AUC: {avg_test_auc:.4f}")

        final_summary = {
            'average_val_loss_scaled': avg_val_loss,
            'average_test_rmse': avg_test_rmse,
            'average_test_correlation': avg_test_pcc,
            'average_test_auc': avg_test_auc,
            'all_fold_validation_results': all_fold_results,
            'all_fold_test_results': all_test_results,
            'parameters': params
        }
    else:
        print("\nSkipping final test results summary as no test data was loaded.")
        final_summary = {
            'average_val_loss_scaled': avg_val_loss,
            'all_fold_validation_results': all_fold_results,
            'parameters': params,
            'note': 'Test data not loaded or found.'
        }

    with open('training_summary_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    print("Training summary saved to training_summary_results.json")

# --- Standard Python Entry Point ---
if __name__ == "__main__":
    main()