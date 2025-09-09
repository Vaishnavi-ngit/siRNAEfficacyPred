import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch.nn import BatchNorm1d, Dropout
from torch.nn import Linear, Dropout, ModuleList, ModuleDict, BatchNorm1d
from torch_geometric.nn import HeteroConv, GATConv
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
import utils  # Features calculation
import optuna

# Load parameters
params = json.load(open("siRNA_param_pytorch.json", 'r'))

# Initialize metric lists
all_fold_pccs = []
class HeteroGAT(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, heads, dropout_rate=0.5, concat=True):
        super().__init__()

        self.convs = ModuleList()
        self.bns = ModuleList()
        self.dropouts = ModuleList()
        self.node_types = metadata[0]  # (node_types, edge_types)
        input_dims: dict = {'siRNA': 1513,'mRNA': 39126,'interaction': 197,}
        self.input_proj = torch.nn.ModuleDict({node_type: Linear(input_dims[node_type], layer_sizes[0])for node_type in self.node_types})
        for i in range(len(layer_sizes)):
            in_channels = layer_sizes[i - 1] if i > 0 else layer_sizes[0]  # assume initial features have layer_sizes[0] dim
            out_channels_i = layer_sizes[i] // heads  # divide for multi-heads

            conv = HeteroConv({('siRNA', 'interacts_with', 'interaction'): GATConv(in_channels, out_channels_i, heads=heads, dropout=dropout_rate, add_self_loops=False),('mRNA', 'interacts_with', 'interaction'): GATConv(in_channels, out_channels_i, heads=heads, dropout=dropout_rate, add_self_loops=False),('interaction', 'rev_interacts_with', 'siRNA'): GATConv(in_channels, out_channels_i, heads=heads, dropout=dropout_rate, add_self_loops=False),('interaction', 'rev_interacts_with', 'mRNA'): GATConv(in_channels, out_channels_i, heads=heads, dropout=dropout_rate, add_self_loops=False),}, aggr='sum')
            self.convs.append(conv)

            self.bns.append(ModuleDict({
                node_type: BatchNorm1d(layer_sizes[i]) for node_type in self.node_types
            }))

            self.dropouts.append(Dropout(dropout_rate))

        self.lin = Linear(layer_sizes[-1], out_channels)

    def forward(self, x_dict, edge_index_dict):
        if self.input_proj:
            x_dict = {node_type: self.input_proj[node_type](x) if node_type in self.input_proj else x
                for node_type, x in x_dict.items()}
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

print(params["dropout"])
print(params["lr"])
print(params["epochs"])
print(params["hinsage_layer_sizes"])
print(params["hop_samples"])
print(params["heads"])
print("learning rate schedule")
print("GAT with concat")


def objective(trial):
    """
    Objective function for Optuna optimization.  It trains and evaluates the model
    across all folds, and returns the mean PCC score.
    """
    score_PCC = []
    score_SPCC = []
    score_mse = []
    score_auc = []

    # Hyperparameter suggestions from Optuna
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.5)
    num_layers = trial.suggest_int("num_layers", 2, 3)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    hop_samples = trial.suggest_categorical("hop_samples", [[10, 5], [15, 10], [20, 15]])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    layer_sizes = [hidden_channels] * num_layers
    epochs = params["epochs"]  # Keep epochs constant
    for n in range(1):
        print(f"Processing fold {n}")
        #print("10")
        # Read and preprocess data
        data_train = pd.read_csv(f"siRNA_split_datasets/split6/train.csv")
        data_dev = pd.read_csv(f"siRNA_split_datasets/split6/dev.csv")
        data_test = pd.read_csv(f"siRNA_split_datasets/split6/test.csv")
    
    
        data_train['split'] = 'train'
        data_dev['split'] = 'dev' 
        data_test['split'] = 'test'
    
    
        # Substitute U to T
        for df in [data_train, data_dev, data_test]:
            df['siRNA_seq'] = df['siRNA_seq'].str.replace('U', 'T')
    
        data = pd.concat([data_train, data_dev, data_test], axis=0)
    
        # Feature processing (same as original)
        '''

        Feature processing

        '''

        # one-hot
        ## siRNA
        sirna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["sirna_length"]) for seq in data['siRNA_seq']]
        sirna_onehot = pd.DataFrame(sirna_onehot,index=list(data['siRNA']))


        ## mRNA
        mrna_onehot_temp = data.loc[:,['mRNA','mRNA_seq']]
        mrna_onehot_temp = mrna_onehot_temp.drop_duplicates(subset="mRNA")

        mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["max_mrna_len"]) for seq in mrna_onehot_temp['mRNA_seq']]
        mrna_onehot = pd.DataFrame(mrna_onehot,index = list(mrna_onehot_temp['mRNA']))


        # Positional encoding
        trans_table = str.maketrans('ATCG', 'TAGC')
        data['match_pos'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
        data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].index(row['match_pos']),axis = 1)


        sirna_pos_encoding = [utils.get_pos_embedding_sequence(num,params["sirna_length"],params["dmodel"]) for num in data['match_pos']]
        sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding,index = list(data['siRNA']))
    

        # Thermodynamics
        sirna_thermo_feat = [utils.cal_thermo_feature(seq.replace("T","U")) for seq in data['siRNA_seq']]

        sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

        sirna_thermo_feat = pd.concat([data['siRNA'].reset_index(drop=True),
                                   data['mRNA'].reset_index(drop=True),
                                   sirna_thermo_feat],
                                   axis = 1)

        sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
        sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])


        # Co-fold features
        con_feat = pd.read_csv("siRNA_split_preprocess/con_matrix.txt",header=None,index_col=0)

        con_feat = con_feat.reindex(sirna_thermo_feat.index)


        # sel-fold features
        ## siRNA
        sirna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_siRNA_matrix.txt",header=None,index_col=0)

        sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)

        ## mRNA
        mrna_sfold_feat = pd.read_csv("siRNA_split_preprocess/self_mRNA_matrix.txt",header=None,index_col=0)

        mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index)


        # AGO2
        ## siRNA-AGO2
        sirna_ago = pd.read_csv("RNA_AGO2/siRNA_AGO2.csv",index_col = 0)

        sirna_ago = sirna_ago.reindex(sirna_onehot.index)

        ## mRNA-AGO2
        mrna_ago = pd.read_csv("RNA_AGO2/mRNA_AGO2.csv",index_col=0)

        mrna_ago = mrna_ago.reindex(mrna_onehot.index)


        # GC percentage
        ## siRNA
        sirna_GC = [utils.countGC(seq) for seq in data['siRNA_seq']]
        sirna_GC = pd.DataFrame(sirna_GC,index=list(data['siRNA']))

        ## mRNA
        mrna_GC = [utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq']]
        mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_onehot_temp['mRNA']))

        # k-mers
        sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data['siRNA_seq']])
        sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data['siRNA_seq']])
        sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data['siRNA_seq']])
        sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
        sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

        sirna_k_mers = pd.concat([sirna_1_mer,sirna_2_mers, sirna_3_mers,sirna_4_mers,sirna_5_mers], axis = 1)
        sirna_k_mers.index = data['siRNA']


        # siRNA rules codes
        sirna_pos_scores = [utils.rules_scores(seq) for seq in data['siRNA_seq']]
        sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index = list(data['siRNA']))
        # ... [include all your feature processing code here] ...
    
        '''

        The features of GNN nodes

        '''

        # siRNA nodes
        sirna_pd = pd.concat([sirna_onehot,sirna_sfold_feat,sirna_ago,sirna_GC,sirna_k_mers,sirna_pos_scores],axis = 1)


        # mRNA nodes
        mrna_pd = pd.concat([mrna_onehot,mrna_sfold_feat,mrna_ago,mrna_GC],axis = 1)


        # edges
        source = data['siRNA'] + "_" + data['mRNA']
        target_siRNA = data['siRNA']
        target_mRNA = data['mRNA']

        all_my_edges1 = pd.DataFrame({'source':source,'target':target_siRNA})
        all_my_edges2 = pd.DataFrame({'source':source,'target':target_mRNA})

        all_my_edges = pd.concat([all_my_edges1,all_my_edges2],ignore_index=True, axis=0)


        # interactive nodes
        sirna_pos_encoding.index = sirna_thermo_feat.index

        interaction_pd = pd.concat([sirna_thermo_feat,con_feat,sirna_pos_encoding],axis=1)
        # Create heterogeneous graph data
        data_hetero = HeteroData()
    
        # Add node features
        data_hetero['siRNA'].x = torch.tensor(sirna_pd.values, dtype=torch.float)
        data_hetero['mRNA'].x = torch.tensor(mrna_pd.values, dtype=torch.float)
        data_hetero['interaction'].x = torch.tensor(interaction_pd.values, dtype=torch.float)
    
        # Add edges
        # Create mapping from node names to indices
        sirna_idx = {name: i for i, name in enumerate(sirna_pd.index)}
        mrna_idx = {name: i for i, name in enumerate(mrna_pd.index)}
        interaction_idx = {name: i for i, name in enumerate(interaction_pd.index)}
    
        # Create edge indices
        edge_index_siRNA_to_interaction = []
        edge_index_mRNA_to_interaction = []
    
        for _, row in data.iterrows():
            interaction_name = f"{row['siRNA']}_{row['mRNA']}"
            edge_index_siRNA_to_interaction.append([sirna_idx[row['siRNA']], interaction_idx[interaction_name]])
            edge_index_mRNA_to_interaction.append([mrna_idx[row['mRNA']], interaction_idx[interaction_name]])
        
    
        data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_siRNA_to_interaction, dtype=torch.long).t().contiguous()
        data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index = torch.tensor(
        edge_index_mRNA_to_interaction, dtype=torch.long).t().contiguous()
        

    
        #print(data_hetero.edge_types)
    
        #print("undirected removed")
        data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

        data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    
        # Split data into train/dev/test
        train_mask = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.bool)
        dev_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)
    
        interaction_idx = {f"{row.siRNA}_{row.mRNA}": i for i, row in enumerate(data.itertuples())}
    
    
        # 4. NOW ADD THE SPLIT INDICES HERE (NEW CODE)
        train_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_train.iterrows()], dtype=torch.long)

        dev_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_dev.iterrows()], dtype=torch.long)

        test_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data_test.iterrows() ], dtype=torch.long)
        #print("removed")
        data_hetero = ToUndirected()(data_hetero)
        #print(data_hetero.node_types)
        #print(data_hetero.edge_types)
        x_dict = data_hetero.x_dict
        edge_index_dict = data_hetero.edge_index_dict

    
        #num_workers=4,
        #persistent_workers=True

        # Try fetching the first batch
 
        train_loader = NeighborLoader(
            data_hetero,
            num_neighbors=params["hop_samples"],
            batch_size=params["batch_size"],
            input_nodes=('interaction', train_idx),
            shuffle=True,
            subgraph_type='induced',
            filter_per_worker=False
        )

        val_loader = NeighborLoader(
            data_hetero,
            num_neighbors=params["hop_samples"],
            batch_size=params["batch_size"],
            input_nodes=('interaction', dev_idx),
            shuffle=False,
            subgraph_type='induced',
            filter_per_worker=False
        )

        test_loader = NeighborLoader(
            data_hetero,
            num_neighbors=params["hop_samples"],
            batch_size=params["batch_size"],
            input_nodes=('interaction', test_idx),
            shuffle=False,
            subgraph_type='induced',
            filter_per_worker=False
        )

    
        train_interactions = set(data_train['siRNA'] + "_" + data_train['mRNA'])
        dev_interactions = set(data_dev['siRNA'] + "_" + data_dev['mRNA'])
        test_interactions = set(data_test['siRNA'] + "_" + data_test['mRNA'])
    
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
    
        # Create labels tensor
        labels = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.float)
        for _, row in data.iterrows():
            interaction_name = f"{row['siRNA']}_{row['mRNA']}"
            labels[interaction_idx[interaction_name]] = row['efficacy']
        data_hetero['interaction'].y = labels
    
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        model = HeteroGAT(
            layer_sizes=params["hinsage_layer_sizes"],
            out_channels=1,
            metadata=data_hetero.metadata(),  # Still needed for BatchNorm
            heads=params["heads"], # Default to 2 heads if not specified
            dropout_rate=params["dropout"]
        ).to(device)

        data_hetero = data_hetero.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)  # L2 via weight_decay
        print("AdamW")
    
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss() if params["loss"] == "mse" else nn.L1Loss()

        def train():
            model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                batch_size = batch['interaction'].num_nodes
                batch_indices = torch.arange(batch_size, device=device)

                out = model(batch.x_dict, batch.edge_index_dict)

                loss = criterion(out[batch_indices].squeeze(-1),batch['interaction'].y[batch_indices])

                # Add L1 regularization
                l1_lambda = params.get("l1_lambda", 1e-5)
                loss += l1_lambda * model.l1_loss()

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            return total_loss / len(train_loader)


        @torch.no_grad()
        def test(loader):
            model.eval()
            preds, truths = [], []
            for batch in loader:
                batch_size = batch['interaction'].num_nodes
                batch_indices = torch.arange(batch_size)
        
        
                out = model(batch.x_dict, batch.edge_index_dict)
                preds.append(out[batch_indices].cpu())
                truths.append(batch['interaction'].y[batch_indices].cpu())
            return torch.cat(preds), torch.cat(truths)
    
        best_val_loss = float('inf')
        for epoch in range(params["epochs"]):
            #print(epoch,"/",params["epochs"])
            loss = train()
        
            # Validation
            val_pred, val_true = test(val_loader)
            #val_loss = mean_squared_error(val_true, val_pred)
            val_loss = F.mse_loss(val_pred.squeeze(-1), val_true).item()
            scheduler.step(val_loss)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold{n}.pt')
        
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
    
    
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(f'best_model_fold{n}.pt'))
        #test_pred, test_true = test(data_hetero['interaction'].test_mask)
        with torch.no_grad():
            test_pred, test_true = test(test_loader)
            test_pred = test_pred.cpu().numpy().flatten()
            test_true = test_true.cpu().numpy().flatten()
    
            # 1. Filter invalid values
            valid_mask = ~np.isnan(test_pred) & ~np.isnan(test_true)
            if not np.any(valid_mask):
                print("Warning: All predictions are NaN!")
                continue
        
            #test_pred = test_pred.flatten()  # Converts (N,1) to (N,)
            #test_true = test_true.flatten()
            test_pred = test_pred[valid_mask]
            test_true = test_true[valid_mask]
    
            # 2. Calculate metrics
                
            r_value,_ = scipy.stats.pearsonr(test_true, test_pred)
            score_PCC.append(r_value)
            #print("bye")
            #print(score_PCC)
            print("PCC:", r_value)
        
            spearman = scipy.stats.spearmanr(test_true, test_pred)
            score_SPCC.append(spearman[0])
            print("SPCC:", spearman[0])
    
    
            score_mse.append(mean_squared_error(test_true, test_pred))
            print(f"Fold {n} finished!  PCC: {score_PCC[-1]:.4f}")
            all_fold_pccs.append(score_PCC[-1])  # Store the PCC for this fold

    
    
            if len(np.unique(test_true > 0.7)) > 1:  # Check if binary labels have variety
                score_auc.append(roc_auc_score((test_true > 0.7).astype(int), test_pred))

    # Calculate the mean PCC across all folds
    mean_pcc = np.mean(score_PCC)
    print(f"Mean PCC across all folds: {mean_pcc:.4f}")
    return mean_pcc  # Optuna minimizes, so return negative PCC
    
    print(f"Fold {n} finished!")

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")  # We want to maximize PCC
    study.optimize(objective, n_trials=20) # Adjust the number of trials as needed.  I've reduced it to 20 for demonstration

    print("Finished Optuna optimization")
    print("Best trial:")
    trial = study.best_trial
    print("  Value (Mean PCC): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Retrain the model with the best hyperparameters on the full dataset
    # (You would typically do this in a separate script or after the optimization)
    #  I am not including the retrain in this script.

