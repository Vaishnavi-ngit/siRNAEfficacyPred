#single value inference
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
import src.utils  # Features calculation

import sys
from flask import request
import subprocess
import os


print("preprocessing")
# Load parameters
params = json.load(open("src/siRNA_param_pytorch .json", 'r'))

print(params['hop_samples'])
print(params['hinsage_layer_sizes'])

class HeteroSAGE(torch.nn.Module):
    def __init__(self, layer_sizes, out_channels, metadata, dropout_rate=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.node_types = metadata[0]  # (node_types, edge_types)

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

            # BatchNorm for each node type
            self.bns.append(nn.ModuleDict({
                node_type: BatchNorm1d(out_channels_i) for node_type in self.node_types
            }))

            self.dropouts.append(Dropout(dropout_rate))

        self.lin = Linear(layer_sizes[-1], out_channels)
        #print("1556")
        #self.lin = Linear(1556, out_channels)

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

def main():
    reqId=request.form.get('reqId')
    # print(reqID)
    #reqId="1234"
    print("Preprocessing........")
    subprocess.run(["python3", "src/preprocess.py",reqId])
    # Load the single CSV file for testing
    print("main entered")
    test_file = f"src/data/input/{reqId}inference_siRNA_mRNA_pairs.csv"  #  CHANGE THIS PATH
    try:
        data = pd.read_csv(test_file)
        print(f"Successfully loaded test data from: {test_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {test_file}. Please check the file path.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        exit()

    # Substitue U to T in siRNA sequence
    data['siRNA_seq'] = data['siRNA_seq'].replace('U', 'T', regex=True)


    '''

    Feature processing

    '''

    # one-hot
    ## siRNA
    sirna_onehot = [src.utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["sirna_length"]) for seq in data['siRNA_seq']]
    sirna_onehot = pd.DataFrame(sirna_onehot,index=list(data['siRNA']))


    ## mRNA
    mrna_onehot_temp = data.loc[:,['mRNA','mRNA_seq']]
    mrna_onehot_temp = mrna_onehot_temp.drop_duplicates(subset="mRNA")

    mrna_onehot = [src.utils.obtain_one_hot_feature_for_one_sequence_1(seq,params["max_mrna_len"]) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_onehot = pd.DataFrame(mrna_onehot,index = list(mrna_onehot_temp['mRNA']))


    # Positional encoding
    trans_table = str.maketrans('ATCG', 'TAGC')
    data['match_pos'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
    data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].index(row['match_pos']),axis = 1)


    sirna_pos_encoding = [src.utils.get_pos_embedding_sequence(num,params["sirna_length"],params["dmodel"]) for num in data['match_pos']]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding,index = list(data['siRNA']))


    # Thermodynamics
    sirna_thermo_feat = [src.utils.cal_thermo_feature(seq.replace("T","U")) for seq in data['siRNA_seq']]

    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)

    sirna_thermo_feat = pd.concat([data['siRNA'].reset_index(drop=True),
                                   data['mRNA'].reset_index(drop=True),
                                   sirna_thermo_feat],
                                   axis = 1)

    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])


    # Co-fold features
    con_feat = pd.read_csv(f"src/data/input/{reqId}preprocess/{reqId}con_matrix_meanSum50.txt", header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index)

    # sel-fold features
    ## siRNA
    sirna_sfold_feat = pd.read_csv(f"src/data/input/{reqId}preprocess/{reqId}self_siRNA_matrix_meanSum6.txt", header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index)

    ## mRNA
    mrna_sfold_feat = pd.read_csv(f"src/data/input/{reqId}preprocess/{reqId}con_matrix_meanSum100.txt", header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index)

    # GC percentage
    ## siRNA
    sirna_GC = [src.utils.countGC(seq) for seq in data['siRNA_seq']]
    sirna_GC = pd.DataFrame(sirna_GC,index=list(data['siRNA']))

    ## mRNA
    mrna_GC = [src.utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_GC = pd.DataFrame(mrna_GC, index=list(mrna_onehot_temp['mRNA']))

    # k-mers
    sirna_1_mer = pd.DataFrame([src.utils.single_freq(seq) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([src.utils.double_freq(seq) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([src.utils.triple_freq(seq) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([src.utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([src.utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

    sirna_k_mers = pd.concat([sirna_1_mer,sirna_2_mers, sirna_3_mers,sirna_4_mers,sirna_5_mers], axis = 1)
    sirna_k_mers.index = data['siRNA']


    # siRNA rules codes
    sirna_pos_scores = [src.utils.rules_scores(seq) for seq in data['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index = list(data['siRNA']))
    # ... [include all your feature processing code here] ...
    
    '''

    The features of GNN nodes

    '''

    # siRNA nodes
    sirna_pd = pd.concat([sirna_onehot,sirna_sfold_feat,sirna_GC,sirna_k_mers,sirna_pos_scores],axis = 1)
    #sirna_pd = pd.concat([sirna_onehot,sirna_sfold_feat,sirna_GC,sirna_k_mers,sirna_pos_scores],axis = 1)


    # mRNA nodes
    mrna_pd = pd.concat([mrna_onehot,mrna_sfold_feat,mrna_GC],axis = 1)
    #mrna_pd = pd.concat([mrna_onehot,mrna_sfold_feat,mrna_GC],axis = 1)


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
    
    #---------------------------------------------------
    
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
        

    
    #print("undirected removed")
    data_hetero['interaction', 'rev_interacts_with', 'siRNA'].edge_index = data_hetero['siRNA', 'interacts_with', 'interaction'].edge_index.flip([0])

    data_hetero['interaction', 'rev_interacts_with', 'mRNA'].edge_index = data_hetero['mRNA', 'interacts_with', 'interaction'].edge_index.flip([0])
    
    # Split data into train/dev/test
    train_mask = torch.zeros(data_hetero['interaction'].num_nodes, dtype=torch.bool)
    dev_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    
    interaction_idx = {f"{row.siRNA}_{row.mRNA}": i for i, row in enumerate(data.itertuples())}
    

    test_idx = torch.tensor([interaction_idx[f"{row['siRNA']}_{row['mRNA']}"] for _, row in data.iterrows() ], dtype=torch.long)

    data_hetero = ToUndirected()(data_hetero)

    x_dict = data_hetero.x_dict
    edge_index_dict = data_hetero.edge_index_dict


    test_loader = NeighborLoader(
    data_hetero,
    num_neighbors=params["hop_samples"],
    batch_size=params["batch_size"],
    input_nodes=('interaction', test_idx),  # Use your test indices
    shuffle=False,
    subgraph_type='induced',
    filter_per_worker=False
)
    

    test_interactions = set(data['siRNA'] + "_" + data['mRNA'])
    
    for i, name in enumerate(interaction_pd.index):

        if name in test_interactions:
            test_mask[i] = True
    
    data_hetero['interaction'].train_mask = train_mask
    data_hetero['interaction'].dev_mask = dev_mask
    data_hetero['interaction'].test_mask = test_mask
    
    

    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HeteroSAGE(
        layer_sizes=params["hinsage_layer_sizes"],
        out_channels=1,
        metadata=data_hetero.metadata(),  # Required for BatchNorm
        dropout_rate=params["dropout"]
    ).to(device)

    data_hetero = data_hetero.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)  # L2 via weight_decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss() if params["loss"] == "mse" else nn.L1Loss()

    # 2. Load the saved weights
    path=f"src/checkpoints/best_model_fold.pt"
    model.load_state_dict(torch.load(path))

    # 3. Set the model to evaluation mode (if evaluating or testing)
    model.eval()
    
    @torch.no_grad()
    def inference(loader):
        model.eval()
        preds = []

        for batch in loader:
            batch = batch.to(device)

            batch_size = batch['interaction'].num_nodes
            batch_indices = torch.arange(batch_size, device=device)

            out = model(batch.x_dict, batch.edge_index_dict)
            preds.append(out[batch_indices].cpu())

        return torch.cat(preds)
        
    inference_preds = inference(test_loader)  # inference_loader = your DataLoader for inference dataset

    # Convert to numpy array
    inference_preds = inference_preds.numpy().flatten()

    print("Predictions:", inference_preds)
 
    return inference_preds
    
if __name__ == "__main__":
    main()
    


