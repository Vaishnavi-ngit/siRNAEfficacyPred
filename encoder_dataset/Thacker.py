import os
import pandas as pd
import numpy as np
import json
import Utils as utils1 # Assuming Utils.py is available for feature calculation functions

# --- Define paths and parameters (consistent with Hill.py) ---
try:
    with open("param.json", 'r') as f:
        params = json.load(f)
except FileNotFoundError:
    print("param.json not found, using default parameters for checks.")
    params = {}

# Ensure these match the values in your param.json / Hill.py
MAX_SIRNA_LENGTH = params.setdefault("sirna_length", 31) 
MAX_MRNA_LENGTH = params.setdefault("max_mrna_len", 2001)
params.setdefault("dmodel", 256) # Needed if using get_pos_embedding_sequence for feature comparison

NUM_FOLDS = 10

# K-fold data paths
KFOLD_SPLIT_DIR_PREFIX = "siRNA_split_datasets/split"

# Test Data Paths (Simone Dataset) - Ensure these are exact as used in Hill.py
TEST_DATA_CSV_PATH = "../encoder_Simone/Simone_all_data.csv"
TEST_SIRNA_SELF_MATRIX_PATH = "../encoder_Simone/Simone_split_preprocess/self_siRNA_matrix_Simone_meanSum6.txt"
TEST_MRNA_SELF_MATRIX_PATH = "../encoder_Simone/Simone_split_preprocess/self_mRNA_matrix.txt"
TEST_CON_MATRIX_PATH = "../encoder_Simone/Simone_split_preprocess/con_matrix.txt"


def run_data_checks():
    print("--- Starting Data Consistency and Leakage Checks (Thacker.py) ---")

    # --- 1. Load All Datasets ---
    print("\n1. Loading Datasets and Collecting IDs/Pairs...")
    
    # Store DataFrames for later use (e.g., efficacy stats)
    kfold_train_dfs = []
    kfold_dev_dfs = []

    # Store sets for efficient overlap checking
    all_kfold_combined_pairs = set() # (siRNA_ID, mRNA_ID) tuples
    all_kfold_unique_siRNAs = set()
    all_kfold_unique_mRNAs = set()
    
    # To check Simone IDs against specific fold's train/dev sets
    kfold_siRNA_ids_by_split = {'train': [], 'dev': []}
    kfold_mRNA_ids_by_split = {'train': [], 'dev': []}


    for n in range(NUM_FOLDS):
        current_split_dir = f"{KFOLD_SPLIT_DIR_PREFIX}{n}/"
        try:
            train_df = pd.read_csv(os.path.join(current_split_dir, "train.csv"))
            dev_df = pd.read_csv(os.path.join(current_split_dir, "dev.csv"))
            
            # Clean sequences (U to T, uppercase) as done in Hill.py for consistency
            for df in [train_df, dev_df]:
                df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T').str.upper()
                df['mRNA_seq_RNA-FM'] = df['mRNA_seq_RNA-FM'].str.upper()

            kfold_train_dfs.append(train_df)
            kfold_dev_dfs.append(dev_df)

            # Collect pairs and unique IDs for overlap checks
            all_kfold_combined_pairs.update(train_df.apply(lambda row: (row['siRNA'], row['mRNA']), axis=1))
            all_kfold_combined_pairs.update(dev_df.apply(lambda row: (row['siRNA'], row['mRNA']), axis=1))
            
            all_kfold_unique_siRNAs.update(train_df['siRNA'].unique())
            all_kfold_unique_siRNAs.update(dev_df['siRNA'].unique())
            all_kfold_unique_mRNAs.update(train_df['mRNA'].unique())
            all_kfold_unique_mRNAs.update(dev_df['mRNA'].unique())

            kfold_siRNA_ids_by_split['train'].append(set(train_df['siRNA'].unique()))
            kfold_siRNA_ids_by_split['dev'].append(set(dev_df['siRNA'].unique()))
            kfold_mRNA_ids_by_split['train'].append(set(train_df['mRNA'].unique()))
            kfold_mRNA_ids_by_split['dev'].append(set(dev_df['mRNA'].unique()))


        except FileNotFoundError:
            print(f"Warning: Data for fold {n} not found at {current_split_dir}. Skipping this fold in checks.")
            continue

    simone_df = None # Initialize to None
    simone_pairs = set()
    simone_unique_siRNAs = set()
    simone_unique_mRNAs = set()

    try:
        simone_df = pd.read_csv(TEST_DATA_CSV_PATH)
        simone_df['siRNA_seq'] = simone_df['siRNA_seq'].str.replace('U', 'T').str.upper()
        simone_df['mRNA_seq_RNA-FM'] = simone_df['mRNA_seq_RNA-FM'].str.upper()
        simone_pairs.update(simone_df.apply(lambda row: (row['siRNA'], row['mRNA']), axis=1))
        simone_unique_siRNAs.update(simone_df['siRNA'].unique())
        simone_unique_mRNAs.update(simone_df['mRNA'].unique())
    except FileNotFoundError:
        print(f"Error: Simone test data not found at '{TEST_DATA_CSV_PATH}'. Test-related checks will be skipped.")

    print(f"\n   --- Summary of Loaded Data ---")
    print(f"   Total interactions across all {NUM_FOLDS} K-fold splits (unique pairs): {len(all_kfold_combined_pairs)}")
    print(f"   Total unique siRNAs across all K-fold splits: {len(all_kfold_unique_siRNAs)}")
    print(f"   Total unique mRNAs across all K-fold splits: {len(all_kfold_unique_mRNAs)}")
    if simone_df is not None:
        print(f"   Total interactions in Simone Test data: {len(simone_df)}")
        print(f"   Total unique siRNAs in Simone Test data: {len(simone_unique_siRNAs)}")
        print(f"   Total unique mRNAs in Simone Test data: {len(simone_unique_mRNAs)}")
    else:
        print("   Simone Test data could not be loaded for detailed checks.")

    # --- 2. Basic Statistics of Efficacy ---
    print("\n2. Efficacy Distribution Statistics:")
    print("   ---------------------------------")
    
    if kfold_train_dfs:
        print("   K-fold Data Efficacy:")
        # For overall K-fold
        all_kfold_efficacy = pd.concat([df['efficacy'] for df in kfold_train_dfs + kfold_dev_dfs])
        print(f"      Overall K-fold (Train+Dev): Mean={all_kfold_efficacy.mean():.4f}, Std={all_kfold_efficacy.std():.4f}, Min={all_kfold_efficacy.min():.4f}, Max={all_kfold_efficacy.max():.4f}")
        
        # Per-fold Efficacy
        for i in range(len(kfold_train_dfs)):
            train_efficacy = kfold_train_dfs[i]['efficacy']
            dev_efficacy = kfold_dev_dfs[i]['efficacy']
            print(f"      Fold {i+1} Train Efficacy: Mean={train_efficacy.mean():.4f}, Std={train_efficacy.std():.4f}")
            print(f"      Fold {i+1} Dev Efficacy: Mean={dev_efficacy.mean():.4f}, Std={dev_efficacy.std():.4f}")
    else:
        print("   No K-fold data loaded for efficacy statistics.")

    if simone_df is not None:
        print("\n   Simone Test Data Efficacy:")
        print(f"      Mean={simone_df['efficacy'].mean():.4f}, Std={simone_df['efficacy'].std():.4f}, Min={simone_df['efficacy'].min():.4f}, Max={simone_df['efficacy'].max():.4f}")
    else:
        print("   Simone Test Data not available for efficacy comparison.")

    # --- 3. Overlap Checks (Leakage Analysis) ---
    print("\n3. Overlap Checks (CRITICAL for Leakage and Generalization):")

    # a. Interaction Pair Overlap
    print("\na. Interaction Pair Overlap:")
    if simone_df is not None:
        common_pairs_kfold_simone = all_kfold_combined_pairs.intersection(simone_pairs)
        print(f"   Overlap between ALL K-fold (Train+Dev) interactions and Simone Test interactions: {len(common_pairs_kfold_simone)} pairs.")
        if len(common_pairs_kfold_simone) > 0:
            print("   >>> WARNING: DIRECT INTERACTION PAIR LEAKAGE DETECTED! This is a major issue compromising test set independence.")
            print(f"   Example overlapping pairs: {list(common_pairs_kfold_simone)[:5]}")
        else:
            print("   No direct interaction pair overlap detected between K-fold data and Simone Test data. (Good)")
        
        # Check dev set specific overlap with Simone
        print("\n   Checking each K-fold Dev set against Simone Test set for interaction pair leakage:")
        for i, dev_df in enumerate(kfold_dev_dfs):
            current_dev_pairs = set(dev_df.apply(lambda row: (row['siRNA'], row['mRNA']), axis=1))
            common_pairs_dev_simone = current_dev_pairs.intersection(simone_pairs)
            if len(common_pairs_dev_simone) > 0:
                print(f"   >>> WARNING: Fold {i+1} Dev set has {len(common_pairs_dev_simone)} interaction pairs overlapping with Simone Test set.")
                print(f"   Example overlapping pairs from Fold {i+1} Dev: {list(common_pairs_dev_simone)[:5]}")
        
        # Check train set specific overlap with Simone
        print("\n   Checking each K-fold Train set against Simone Test set for interaction pair leakage:")
        for i, train_df in enumerate(kfold_train_dfs):
            current_train_pairs = set(train_df.apply(lambda row: (row['siRNA'], row['mRNA']), axis=1))
            common_pairs_train_simone = current_train_pairs.intersection(simone_pairs)
            if len(common_pairs_train_simone) > 0:
                print(f"   >>> WARNING: Fold {i+1} Train set has {len(common_pairs_train_simone)} interaction pairs overlapping with Simone Test set.")
                print(f"   Example overlapping pairs from Fold {i+1} Train: {list(common_pairs_train_simone)[:5]}")
    else:
        print("   Simone Test data not available for interaction pair overlap checks.")

    # b. Individual Molecule ID Overlap (more subtle leakage / generalization test)
    print("\nb. Individual Molecule ID Overlap:")
    if simone_df is not None:
        common_siRNAs = all_kfold_unique_siRNAs.intersection(simone_unique_siRNAs)
        common_mRNAs = all_kfold_unique_mRNAs.intersection(simone_unique_mRNAs)

        print(f"   Number of siRNAs present in both K-fold data AND Simone Test data: {len(common_siRNAs)}")
        if len(common_siRNAs) > 0:
            print("   Note: Overlapping individual molecule IDs (siRNAs) are common if the test set is 'in-distribution' but just new interactions. This is okay if your goal is interaction-level generalization with seen molecules, but not for molecule-level generalization (new molecules).")
        
        print(f"   Number of mRNAs present in both K-fold data AND Simone Test data: {len(common_mRNAs)} mRNAs.")
        if len(common_mRNAs) > 0:
            print("   Note: Overlapping individual molecule IDs (mRNAs) are common if the test set is 'in-distribution'.")

        # c. Critical ID Leakage: Are Simone Test IDs present *only* in K-fold Train sets?
        print("\nc. IDs in Simone Test that were *exclusively in K-fold Train sets* (not in their respective Dev sets):")
        # This checks if an ID in Simone test data was part of a K-fold training set, 
        # but was NOT part of that fold's corresponding development set (meaning it was seen during training).
        
        test_siRNAs_seen_in_kfold_train_only = set()
        test_mRNAs_seen_in_kfold_train_only = set()

        for i in range(NUM_FOLDS):
            if i < len(kfold_siRNA_ids_by_split['train']): # Ensure fold data was loaded
                train_siRNAs = kfold_siRNA_ids_by_split['train'][i]
                dev_siRNAs = kfold_siRNA_ids_by_split['dev'][i]
                train_mRNAs = kfold_mRNA_ids_by_split['train'][i]
                dev_mRNAs = kfold_mRNA_ids_by_split['dev'][i]

                # IDs in this fold's train set that are NOT in this fold's dev set
                siRNAs_only_in_this_train = train_siRNAs - dev_siRNAs
                mRNAs_only_in_this_train = train_mRNAs - dev_mRNAs

                # Overlap of these "train-only" IDs with Simone test set
                overlap_siRNA = siRNAs_only_in_this_train.intersection(simone_unique_siRNAs)
                overlap_mRNA = mRNAs_only_in_this_train.intersection(simone_unique_mRNAs)

                if overlap_siRNA:
                    test_siRNAs_seen_in_kfold_train_only.update(overlap_siRNA)
                    print(f"   Fold {i+1}: {len(overlap_siRNA)} Simone siRNAs were exclusively in THIS FOLD's TRAINING set (not its DEV set).")
                if overlap_mRNA:
                    test_mRNAs_seen_in_kfold_train_only.update(overlap_mRNA)
                    print(f"   Fold {i+1}: {len(overlap_mRNA)} Simone mRNAs were exclusively in THIS FOLD's TRAINING set (not its DEV set).")
        
        if test_siRNAs_seen_in_kfold_train_only or test_mRNAs_seen_in_kfold_train_only:
            print(f"   >>> IMPORTANT: {len(test_siRNAs_seen_in_kfold_train_only)} Simone siRNAs and {len(test_mRNAs_seen_in_kfold_train_only)} Simone mRNAs were seen *only* in training sets across folds.")
            print("   This indicates that your 'test set' might not be testing generalization to *unseen molecules*, but rather *unseen interactions with seen molecules*. This is not ideal if the goal is molecule-level generalization.")
        else:
            print("   No Simone Test IDs were exclusively present in any K-fold Training set (good for molecule-level generalization).")

    else:
        print("   Simone Test data not available for individual ID overlap checks.")


    # --- 4. Feature Distribution Comparison (Initial check of efficacy and sequence lengths) ---
    print("\n4. Feature Distribution Comparison (Efficacy and Sequence Lengths):")
    
    # Efficacy distribution already done in section 2.
    print("\n   Sequence Lengths:")
    
    # Representative K-fold (e.g., from first fold's combined data)
    if kfold_train_dfs:
        rep_kfold_siRNA_lengths = pd.concat([df['siRNA_seq'].str.len() for df in kfold_train_dfs + kfold_dev_dfs])
        rep_kfold_mRNA_lengths = pd.concat([df['mRNA_seq_RNA-FM'].str.len() for df in kfold_train_dfs + kfold_dev_dfs])
        print(f"   K-fold Combined (Train+Dev) siRNA Seq Lengths: Mean={rep_kfold_siRNA_lengths.mean():.2f}, Std={rep_kfold_siRNA_lengths.std():.2f}, Min={rep_kfold_siRNA_lengths.min()}, Max={rep_kfold_siRNA_lengths.max()}")
        print(f"   K-fold Combined (Train+Dev) mRNA Seq Lengths: Mean={rep_kfold_mRNA_lengths.mean():.2f}, Std={rep_kfold_mRNA_lengths.std():.2f}, Min={rep_kfold_mRNA_lengths.min()}, Max={rep_kfold_mRNA_lengths.max()}")
    
    if simone_df is not None:
        simone_siRNA_lengths = simone_df['siRNA_seq'].str.len()
        simone_mRNA_lengths = simone_df['mRNA_seq_RNA-FM'].str.len()
        print(f"   Simone Test siRNA Seq Lengths: Mean={simone_siRNA_lengths.mean():.2f}, Std={simone_siRNA_lengths.std():.2f}, Min={simone_siRNA_lengths.min()}, Max={simone_siRNA_lengths.max()}")
        print(f"   Simone Test mRNA Seq Lengths: Mean={simone_mRNA_lengths.mean():.2f}, Std={simone_mRNA_lengths.std():.2f}, Min={simone_mRNA_lengths.min()}, Max={simone_mRNA_lengths.max()}")

    # To do a deeper feature distribution comparison (e.g., GC content, thermodynamic features):
    # You would need to re-implement parts of the feature generation from Hill.py here
    # (using utils1 functions) and calculate statistics on those generated features for comparison.
    # This might require a significant amount of code duplication if you want to inspect many features.
    # For initial check, efficacy and sequence lengths are usually good indicators of distribution shift.

    print("\n--- Data Checks Complete ---")

if __name__ == "__main__":
    run_data_checks()