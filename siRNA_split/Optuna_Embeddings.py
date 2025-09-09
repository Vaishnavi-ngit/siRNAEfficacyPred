import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.transforms import ToUndirected
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import scipy.stats
import json
import math
import re
import utils # Features calculation
import optuna # Import Optuna

# We'll load params in the objective function, but keep it here for reference
# params = json.load(open("siRNA_param_pytorch.json", 'r'))

class HeteroSAGE(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0]

        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else -1
            out_channels_i = layer_sizes[i]

            conv = HeteroConv({
                ('siRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('mRNA', 'interacts_with', 'interaction'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'siRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
                ('interaction', 'rev_interacts_with', 'mRNA'): SAGEConv((in_channels, in_channels), out_channels_i),
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
            for node_type in x_dict:
                if node_type in bn_dict:
                    x_dict[node_type] = bn_dict[node_type](x_dict[node_type])
                x_dict[node_type] = F.leaky_relu(x_dict[node_type])
                x_dict[node_type] = dropout(x_dict[node_type])
        return self.lin(x_dict['interaction'])

    def l1_loss(self):
        return sum(p.abs().sum() for p in self.parameters())

# --- Optuna Integration ---

def objective(trial):
    # Load default parameters as a base, then override with trial suggestions
    base_params = json.load(open("siRNA_param_pytorch.json", 'r'))
    
    # 1. Hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.4, 0.8)
    l1_lambda = trial.suggest_loguniform('l1_lambda', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    num_layers = trial.suggest_int('num_layers', 2, 4)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    hinsage_layer_sizes = [hidden_dim] * num_layers

    num_neighbors_hop1 = trial.suggest_int('num_neighbors_hop1', 5, 20)
    num_neighbors_hop2 = trial.suggest_int('num_neighbors_hop2', 3, 10)
    hop_samples = [num_neighbors_hop1, num_neighbors_hop2]

    # Update params dictionary for this trial
    trial_params = {
        **base_params, # Start with base parameters
        "lr": lr,
        "dropout": dropout,
        "batch_size": batch_size,
        "hinsage_layer_sizes": hinsage_layer_sizes,
        "hop_samples": hop_samples,
        "l1_lambda": l1_lambda # Add l1_lambda to params
    }
    
    # --- Data Loading and Preprocessing (remains largely the same) ---
    # We'll run for a single fold for tuning to save time.
    # You can loop over multiple folds here if you need a more robust validation for tuning,
    # but it will significantly increase the tuning time.
    
    # For initial tuning, let's just use fold 0
    n = 0 # Consider running Optuna on a single representative fold (e.g., fold 0)
          # For a more robust tuning, you might average across a few folds or run an outer K-Fold loop.
          # For simplicity here, we stick to one fixed fold for the objective.

    print(f"Processing fold {n} for Optuna trial {trial.number}")
    data_train = pd.read_csv(f"siRNA_split_datasets/split{n}/train.csv")
    data_dev = pd.read_csv(f"siRNA_split_datasets/split{n}/dev.csv")
    data_test = pd.read_csv(f"siRNA_split_datasets/split{n}/test.csv")

    data_train['split'] = 'train'
    data_dev['split'] = 'dev'
    data_test['split'] = 'test'

    for df in [data_train, data_dev, data_test]:
        df['siRNA_seq'] = df['siRNA_seq'].str.replace('T', 'U', regex=False)
        df['mRNA_seq'] = df['mRNA_seq'].str.replace('T', 'U', regex=False)

    data = pd.concat([data_train, data_dev, data_test], axis=0)

    # --- Sequence Embedding using MP-RNA Transformer ---
    sirna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in data['siRNA_seq']]
    sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(data['siRNA']))
    sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')]

    mrna_unique_seq_df = data.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
    mrna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
    mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')]
    # --- End of Sequence Embedding Update ---

    # Positional encoding
    trans_table = str.maketrans('AUCG', 'UAGC')
    data['match_pos_siRNA_rev_comp'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
    data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].find(row['match_pos_siRNA_rev_comp']), axis=1)

    # Generate positional encoding for each interaction, indexed by interaction ID
    # Use trial_params["dmodel"] if it's ever going to be tunable, otherwise base_params["dmodel"]
    sirna_pos_encoding_per_interaction = [utils.get_pos_embedding_sequence(num,trial_params["sirna_length"],trial_params["dmodel"]) for num in data['match_pos']]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding_per_interaction, index = data['siRNA'] + '_' + data['mRNA'])
    sirna_pos_encoding = sirna_pos_encoding.loc[~sirna_pos_encoding.index.duplicated(keep='first')]

    # Thermodynamics
    sirna_thermo_feat = [utils.cal_thermo_feature(seq) for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    sirna_thermo_feat = pd.concat([data['siRNA'].reset_index(drop=True),
                                    data['mRNA'].reset_index(drop=True),
                                    sirna_thermo_feat],
                                    axis = 1)
    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])

    # Co-fold features
    con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt",header=None,index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)

    # self-fold features
    sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt",header=None,index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0)

    mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt",header=None,index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0)

    # GC percentage
    sirna_GC = [utils.countGC(seq) for seq in data['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC,index=list(data['siRNA']))
    sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')]

    mrna_GC = [utils.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_unique_seq_df['mRNA']))
    mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')]

    # k-mers
    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

    sirna_k_mers = pd.concat([sirna_1_mer,sirna_2_mers, sirna_3_mers,sirna_4_mers,sirna_5_mers], axis = 1)
    sirna_k_mers.index = data['siRNA']
    sirna_k_mers = sirna_k_mers.loc[~sirna_k_mers.index.duplicated(keep='first')]

    # siRNA rules codes
    sirna_pos_scores = [utils.rules_scores(seq) for seq in data['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index = list(data['siRNA']))
    sirna_pos_scores = sirna_pos_scores.loc[~sirna_pos_scores.index.duplicated(keep='first')]

    # The features of GNN nodes
    sirna_pd = pd.concat([
        sirna_embedding_df,
        sirna_sfold_feat,
        sirna_GC.reindex(sirna_embedding_df.index).fillna(0),
        sirna_k_mers.reindex(sirna_embedding_df.index).fillna(0),
        sirna_pos_scores.reindex(sirna_embedding_df.index).fillna(0)
    ], axis = 1)
    sirna_pd = sirna_pd.fillna(0)

    mrna_pd = pd.concat([
        mrna_embedding_df,
        mrna_sfold_feat,
        mrna_GC.reindex(mrna_embedding_df.index).fillna(0)
    ], axis = 1)
    mrna_pd = mrna_pd.fillna(0)

    # edges
    source = data['siRNA'] + "_" + data['mRNA']
    target_siRNA = data['siRNA']
    target_mRNA = data['mRNA']

    all_my_edges1 = pd.DataFrame({'source':source,'target':target_siRNA})
    all_my_edges2 = pd.DataFrame({'source':source,'target':target_mRNA})

    all_my_edges = pd.concat([all_my_edges1,all_my_edges2],ignore_index=True, axis=0)

    # interactive nodes
    # Ensure sirna_pos_encoding is correctly indexed by interaction ID
    interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding.reindex(sirna_thermo_feat.index).fillna(0)],axis=1)
    interaction_pd = interaction_pd.fillna(0)

    # Create heterogeneous graph data
    data_hetero = HeteroData()

    # Add node features
    data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
    data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
    data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)

    # Create mapping from node names to indices
    sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
    mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
    interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}

    # Create edge indices
    edge_index_siRNA_to_interaction = []
    edge_index_mRNA_to_interaction = []

    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_idx:
            edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])

    data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
    data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

    data_hetero = ToUndirected()(data_hetero)

    x_dict = data_hetero.x_dict
    edge_index_dict = data_hetero.edge_index_dict

    # Define split indices for NeighborLoader
    train_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)
    dev_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)
    test_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_test.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)

    train_loader = NeighborLoader(data_hetero, num_neighbors=trial_params["hop_samples"], batch_size=trial_params["batch_size"], input_nodes=('interaction', train_idx), shuffle=True, subgraph_type='induced', filter_per_worker=False)

    val_loader = NeighborLoader(
        data_hetero,
        num_neighbors=trial_params["hop_samples"],
        batch_size=trial_params["batch_size"],
        input_nodes=('interaction', dev_idx),
        shuffle=False,
        subgraph_type='induced',
        filter_per_worker=False
    )

    test_loader = NeighborLoader(
        data_hetero,
        num_neighbors=trial_params["hop_samples"],
        batch_size=trial_params["batch_size"],
        input_nodes=('interaction', test_idx),
        shuffle=False,
        subgraph_type='induced',
        filter_per_worker=False
    )

    train_interactions = set(data_train['siRNA'] + "_" + data_train['mRNA'])
    dev_interactions = set(data_dev['siRNA'] + "_" + data_dev['mRNA'])
    test_interactions = set(data_test['siRNA'] + "_" + data_test['mRNA'])

    train_mask = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.bool)
    dev_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    for i, name in enumerate(interaction_pd.index):
        if name in train_interactions:
            train_mask[i] = True
        elif name in dev_interactions:
            dev_mask[i] = True
        elif name in test_interactions:
            test_mask[i] = True

    data_hetero['interaction'].train_mask = train_mask
    data_hetero['interaction'].dev_mask = dev_mask
    data_hetero['interaction'].test_mask = test_mask

    labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
    for _, row in data.iterrows():
        interaction_name = f"{row['siRNA']}_{row['mRNA']}"
        if interaction_name in interaction_idx:
            labels[interaction_idx[interaction_name]] = row['efficacy']
    data_hetero['interaction'].y = labels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HeteroSAGE(
        layer_sizes=trial_params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),
        dropout_rate=trial_params["dropout"]
    ).to(device)

    data_hetero = data_hetero.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_params["lr"], weight_decay=1e-4) # weight_decay is L2, currently fixed.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) # Set verbose to False for cleaner Optuna output
    criterion = nn.MSELoss() if trial_params["loss"] == "mse" else nn.L1Loss()

    def train_step():
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_size_interaction_nodes = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size_interaction_nodes, device=device)
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = criterion(out[batch_indices].squeeze(-1), batch['interaction'].y[batch_indices])
            
            # Apply L1 regularization using the suggested l1_lambda
            loss += trial_params["l1_lambda"] * model.l1_loss()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def test_step(loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            batch_size_interaction_nodes = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size_interaction_nodes) # No need for device here, will move to cpu later
            out = model(batch.x_dict, batch.edge_index_dict)
            preds.append(out[batch_indices].cpu())
            truths.append(batch['interaction'].y[batch_indices].cpu())
        return torch.cat(preds), torch.cat(truths)

    best_val_loss = float('inf')
    early_stopping_patience = 10 # You can tune this or keep it fixed
    epochs_no_improve = 0

    for epoch in range(trial_params["epochs"]):
        train_loss = train_step()

        val_pred, val_true = test_step(val_loader)
        val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()

        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Optionally save the best model state_dict here, but not necessary for Optuna
            # torch.save(model.state_dict(), f'best_model_trial_{trial.number}_fold_{n}.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                # print(f"Early stopping at epoch {epoch} for trial {trial.number}")
                break
        
        # Optuna pruning: Report intermediate value to Optuna
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # After training (possibly with early stopping), load best model if saved, or just return best_val_loss
    # If you saved the model, load it here before final test.
    # If not saved, best_val_loss already reflects the best.
    return best_val_loss

# --- Main Optuna Study Execution ---
if __name__ == "__main__":
    # Create an Optuna study. You can specify the sampler (e.g., TPESampler, CmaEsSampler)
    # and pruner (e.g., MedianPruner) based on your needs.
    # Direction='minimize' because we want to minimize validation MSE.
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30))
    
    # Optimize the objective function. N_trials is the number of different hyperparameter
    # combinations Optuna will try.
    study.optimize(objective, n_trials=50, timeout=7200) # e.g., 50 trials or 2 hours timeout

    print("\n--- Optuna Study Results ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can also access all trials and their results
    # for i, t in enumerate(study.trials):
    #     print(f"Trial {i} finished with value: {t.value} and params: {t.params}")

    # --- Optional: Run final evaluation with best parameters ---
    # After finding the best parameters, you'd typically run your full K-Fold cross-validation
    # with these optimal parameters on your entire dataset to get the final model performance.
    
    print("\nRunning final evaluation with best parameters across all 10 folds:")
    final_params = json.load(open("siRNA_param_pytorch.json", 'r'))
    # Update default params with the best found by Optuna
    for key, value in trial.params.items():
        if key in ['num_layers', 'hidden_dim', 'num_neighbors_hop1', 'num_neighbors_hop2']:
            # Handle special cases for layer_sizes and hop_samples reconstruction
            if key == 'num_layers':
                final_params['hinsage_layer_sizes'] = [trial.params['hidden_dim']] * value
            elif key == 'hidden_dim' and 'num_layers' not in trial.params: # if hidden_dim was tuned but num_layers wasn't part of *this* trial
                final_params['hinsage_layer_sizes'] = [value] * len(final_params['hinsage_layer_sizes'])
            elif key == 'num_neighbors_hop1':
                final_params['hop_samples'][0] = value
            elif key == 'num_neighbors_hop2':
                final_params['hop_samples'][1] = value
        else:
            final_params[key] = value

    # Ensure hinsage_layer_sizes and hop_samples are correctly reconstructed even if only parts were tuned
    if 'num_layers' in trial.params and 'hidden_dim' in trial.params:
        final_params['hinsage_layer_sizes'] = [trial.params['hidden_dim']] * trial.params['num_layers']
    elif 'hidden_dim' in trial.params and 'num_layers' not in trial.params:
        # If hidden_dim was tuned but num_layers was not, apply it to existing num_layers
        final_params['hinsage_layer_sizes'] = [trial.params['hidden_dim']] * len(final_params['hinsage_layer_sizes'])
    
    if 'num_neighbors_hop1' in trial.params and 'num_neighbors_hop2' in trial.params:
        final_params['hop_samples'] = [trial.params['num_neighbors_hop1'], trial.params['num_neighbors_hop2']]
    elif 'num_neighbors_hop1' in trial.params: # if only one was tuned
        final_params['hop_samples'][0] = trial.params['num_neighbors_hop1']
    elif 'num_neighbors_hop2' in trial.params: # if only one was tuned
        final_params['hop_samples'][1] = trial.params['num_neighbors_hop2']

    score_PCC = []
    score_SPCC = []
    score_mse = []
    score_auc = []

    for n in range(10): # Run for all 10 folds
        print(f"\n--- Final Evaluation: Processing fold {n} with best parameters ---")
        # Reuse the data loading and processing logic from the objective function
        data_train = pd.read_csv(f"siRNA_split_datasets/split{n}/train.csv")
        data_dev = pd.read_csv(f"siRNA_split_datasets/split{n}/dev.csv")
        data_test = pd.read_csv(f"siRNA_split_datasets/split{n}/test.csv")

        data_train['split'] = 'train'
        data_dev['split'] = 'dev'
        data_test['split'] = 'test'

        for df in [data_train, data_dev, data_test]:
            df['siRNA_seq'] = df['siRNA_seq'].str.replace('T', 'U', regex=False)
            df['mRNA_seq'] = df['mRNA_seq'].str.replace('T', 'U', regex=False)

        data = pd.concat([data_train, data_dev, data_test], axis=0)

        sirna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in data['siRNA_seq']]
        sirna_embedding_df = pd.DataFrame(sirna_transformer_embeddings, index=list(data['siRNA']))
        sirna_embedding_df = sirna_embedding_df.loc[~sirna_embedding_df.index.duplicated(keep='first')]

        mrna_unique_seq_df = data.loc[:,['mRNA','mRNA_seq']].drop_duplicates(subset="mRNA")
        mrna_transformer_embeddings = [utils.get_mp_rna_sequence_embedding(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
        mrna_embedding_df = pd.DataFrame(mrna_transformer_embeddings, index = list(mrna_unique_seq_df['mRNA']))
        mrna_embedding_df = mrna_embedding_df.loc[~mrna_embedding_df.index.duplicated(keep='first')]

        trans_table = str.maketrans('AUCG', 'UAGC')
        data['match_pos_siRNA_rev_comp'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
        data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].find(row['match_pos_siRNA_rev_comp']), axis=1)

        sirna_pos_encoding_per_interaction = [utils.get_pos_embedding_sequence(num,final_params["sirna_length"],final_params["dmodel"]) for num in data['match_pos']]
        sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding_per_interaction, index = data['siRNA'] + '_' + data['mRNA'])
        sirna_pos_encoding = sirna_pos_encoding.loc[~sirna_pos_encoding.index.duplicated(keep='first')]

        sirna_thermo_feat = [utils.cal_thermo_feature(seq) for seq in data['siRNA_seq']]
        sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
        sirna_thermo_feat = pd.concat([data['siRNA'].reset_index(drop=True),
                                        data['mRNA'].reset_index(drop=True),
                                        sirna_thermo_feat],
                                        axis = 1)
        sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
        sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])

        con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt",header=None,index_col=0)
        con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0)

        sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt",header=None,index_col=0)
        sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_embedding_df.index).fillna(0)

        mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt",header=None,index_col=0)
        mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_embedding_df.index).fillna(0)

        sirna_GC = [utils.countGC(seq) for seq in data['siRNA_seq']]
        sirna_GC = pd.DataFrame(sirna_GC,index=list(data['siRNA']))
        sirna_GC = sirna_GC.loc[~sirna_GC.index.duplicated(keep='first')]

        mrna_GC = [utils.countGC(seq) for seq in mrna_unique_seq_df['mRNA_seq']]
        mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_unique_seq_df['mRNA']))
        mrna_GC = mrna_GC.loc[~mrna_GC.index.duplicated(keep='first')]

        sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data['siRNA_seq']])
        sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data['siRNA_seq']])
        sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data['siRNA_seq']])
        sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
        sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

        sirna_k_mers = pd.concat([sirna_1_mer,sirna_2_mers, sirna_3_mers,sirna_4_mers,sirna_5_mers], axis = 1)
        sirna_k_mers.index = data['siRNA']
        sirna_k_mers = sirna_k_mers.loc[~sirna_k_mers.index.duplicated(keep='first')]

        sirna_pos_scores = [utils.rules_scores(seq) for seq in data['siRNA_seq']]
        sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index = list(data['siRNA']))
        sirna_pos_scores = sirna_pos_scores.loc[~sirna_pos_scores.index.duplicated(keep='first')]

        sirna_pd = pd.concat([
            sirna_embedding_df,
            sirna_sfold_feat,
            sirna_GC.reindex(sirna_embedding_df.index).fillna(0),
            sirna_k_mers.reindex(sirna_embedding_df.index).fillna(0),
            sirna_pos_scores.reindex(sirna_embedding_df.index).fillna(0)
        ], axis = 1)
        sirna_pd = sirna_pd.fillna(0)

        mrna_pd = pd.concat([
            mrna_embedding_df,
            mrna_sfold_feat,
            mrna_GC.reindex(mrna_embedding_df.index).fillna(0)
        ], axis = 1)
        mrna_pd = mrna_pd.fillna(0)

        source = data['siRNA'] + "_" + data['mRNA']
        target_siRNA = data['siRNA']
        target_mRNA = data['mRNA']

        all_my_edges1 = pd.DataFrame({'source':source,'target':target_siRNA})
        all_my_edges2 = pd.DataFrame({'source':source,'target':target_mRNA})

        all_my_edges = pd.concat([all_my_edges1,all_my_edges2],ignore_index=True, axis=0)

        interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding.reindex(sirna_thermo_feat.index).fillna(0)],axis=1)
        interaction_pd = interaction_pd.fillna(0)

        data_hetero = HeteroData()

        data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
        data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
        data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)

        sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
        mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
        interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}

        edge_index_siRNA_to_interaction = []
        edge_index_mRNA_to_interaction = []

        for _, row in data.iterrows():
            interaction_name = f"{row['siRNA']}_{row['mRNA']}"
            if interaction_name in interaction_idx:
                edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
                edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])

        data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
            edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
        data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
            edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()

        data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
        data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

        data_hetero = ToUndirected()(data_hetero)

        x_dict = data_hetero.x_dict
        edge_index_dict = data_hetero.edge_index_dict

        train_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)
        dev_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)
        test_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_test.iterrows() if f"{row['siRNA']}_{row['mRNA']}" in interaction_idx], dtype=torch.long)

        train_loader = NeighborLoader(data_hetero,num_neighbors=final_params["hop_samples"],batch_size=final_params["batch_size"],input_nodes=('interaction', train_idx),shuffle=True, subgraph_type='induced',filter_per_worker=False )

        val_loader = NeighborLoader(
            data_hetero,
            num_neighbors=final_params["hop_samples"],
            batch_size=final_params["batch_size"],
            input_nodes=('interaction', dev_idx),
            shuffle=False,
            subgraph_type='induced',
            filter_per_worker=False
        )

        test_loader = NeighborLoader(
            data_hetero,
            num_neighbors=final_params["hop_samples"],
            batch_size=final_params["batch_size"],
            input_nodes=('interaction', test_idx),
            shuffle=False,
            subgraph_type='induced',
            filter_per_worker=False
        )

        train_interactions = set(data_train['siRNA'] + "_" + data_train['mRNA'])
        dev_interactions = set(data_dev['siRNA'] + "_" + data_dev['mRNA'])
        test_interactions = set(data_test['siRNA'] + "_" + data_test['mRNA'])

        train_mask = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.bool)
        dev_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)

        for i, name in enumerate(interaction_pd.index):
            if name in train_interactions:
                train_mask[i] = True
            elif name in dev_interactions:
                dev_mask[i] = True
            elif name in test_interactions:
                test_mask[i] = True

        data_hetero['interaction'].train_mask = train_mask
        data_hetero['interaction'].dev_mask = dev_mask
        data_hetero['interaction'].test_mask = test_mask

        labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
        for _, row in data.iterrows():
            interaction_name = f"{row['siRNA']}_{row['mRNA']}"
            if interaction_name in interaction_idx:
                labels[interaction_idx[interaction_name]] = row['efficacy']
        data_hetero['interaction'].y = labels

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = HeteroSAGE(
            layer_sizes=final_params["hinsage_layer_sizes"],
            out_channels=1,
            metadata=data_hetero.metadata(),
            dropout_rate=final_params["dropout"]
        ).to(device)

        data_hetero = data_hetero.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=final_params["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        criterion = nn.MSELoss() if final_params["loss"] == "mse" else nn.L1Loss() # Use final_params for loss too

        # train_step and test_step functions are already defined in the objective,
        # but redefine them here to use `final_params` and ensure they are local to this loop.
        def train_step_final():
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                batch_size_interaction_nodes = batch['interaction'].num_nodes
                batch_indices = torch.arange(batch_size_interaction_nodes, device=device)
                out = model(batch.x_dict, batch.edge_index_dict)
                loss = criterion(out[batch_indices].squeeze(-1), batch['interaction'].y[batch_indices])
                
                loss += final_params["l1_lambda"] * model.l1_loss() # Use final_params for l1_lambda

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_loader)

        @torch.no_grad()
        def test_step_final(loader):
            model.eval()
            preds, truths = [], []
            for batch in loader:
                batch_size_interaction_nodes = batch['interaction'].num_nodes
                batch_indices = torch.arange(batch_size_interaction_nodes)
                out = model(batch.x_dict, batch.edge_index_dict)
                preds.append(out[batch_indices].cpu())
                truths.append(batch['interaction'].y[batch_indices].cpu())
            return torch.cat(preds), torch.cat(truths)

        best_val_loss = float('inf')
        early_stopping_patience_final = 10 # You can adjust this for final run
        epochs_no_improve = 0

        for epoch in range(final_params["epochs"]):
            train_loss = train_step_final()

            val_pred, val_true = test_step_final(val_loader)
            val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'best_model_fold{n}.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience_final:
                    print(f"Early stopping at epoch {epoch} for fold {n}")
                    break
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        model.load_state_dict(torch.load(f'best_model_fold{n}.pt'))
        with torch.no_grad():
            test_pred, test_true = test_step_final(test_loader)
            test_pred = test_pred.cpu().numpy().flatten()
            test_true = test_true.cpu().numpy().flatten()

            valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
            if not np.any(valid_mask):
                print("Warning: All predictions are NaN for this fold!")
                continue

            test_pred = test_pred[valid_mask]
            test_true = test_true[valid_mask]

            r_value,_ = scipy.stats.pearsonr(test_true, test_pred)
            score_PCC.append(r_value)
            print("PCC:", r_value)

            spearman = scipy.stats.spearmanr(test_true, test_pred)
            score_SPCC.append(spearman[0])
            print("SPCC:", spearman[0])

            score_mse.append(mean_squared_error(test_true, test_pred))

            if len(np.unique(test_true > 0.7)) > 1:
                score_auc.append(roc_auc_score((test_true > 0.7).astype(int), test_pred))

        if score_PCC: # Check if scores were added for this fold
            print(f"\nFold Metrics for fold {n}:")
            print(f"PCC: {score_PCC[-1]:.4f}")
            print(f"SPCC: {score_SPCC[-1]:.4f}")
            print(f"MSE: {score_mse[-1]:.4f}")
            if score_auc and len(score_auc) > len(score_mse)-1: # ensure AUC was calculated
                print(f"AUC: {score_auc[-1]:.4f}")

    print("\n--- Overall Metrics with Best Parameters ---")
    print("Overall PCC score =", np.mean(score_PCC))
    print("Overall SPCC score =", np.mean(score_SPCC))
    print("Overall MSE score =", np.mean(score_mse))
    if score_auc:
        print("Overall AUC score =", np.mean(score_auc))