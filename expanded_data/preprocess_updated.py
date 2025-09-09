from Bio import SeqIO
import pandas as pd
import subprocess
import itertools
import os
import glob
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import cupy as cp # CuPy library imported

# Read mrna file
mrna_file = f"full_mRNA.fas"
mrna_map = []
for seq_record in SeqIO.parse(mrna_file,"fasta"):
    mrna_id = seq_record.id
    seq = str(seq_record.seq).upper().strip()
    mrna_map.append((mrna_id, str(seq)))

mrna_df = pd.DataFrame(mrna_map, columns=['mRNA', 'mRNA_seq'])


# Read sirna file
sirna_file = f"full_siRNA.fas"  # assuming FASTA format for siRNAs
sirna_map = []
for seq_record in SeqIO.parse(sirna_file, "fasta"):
    sirna_id = seq_record.id
    seq = str(seq_record.seq).upper().strip()
    sirna_map.append((sirna_id, seq))
sirna_df = pd.DataFrame(sirna_map, columns=['siRNA', 'siRNA_seq'])

# # Create **all possible siRNA–mRNA pairs** for inference
# pairs_list = list(itertools.product(sirna_df.itertuples(index=False),
#                                      mrna_df.itertuples(index=False)))

# # Prepare pairs DataFrame
# pair_data = []
# for sirna_row, mrna_row in pairs_list:
#     pair_data.append({
#         'siRNA': sirna_row.siRNA,
#         'siRNA_seq': sirna_row.siRNA_seq,
#         'mRNA': mrna_row.mRNA,
#         'mRNA_seq': mrna_row.mRNA_seq,
#         'seq_pairs': sirna_row.siRNA_seq + '&' + mrna_row.mRNA_seq
#     })

# pairs_df = pd.DataFrame(pair_data)

# Combine mRNA & siRNA file
efiicacy_file = pd.read_csv("full_efficacy.csv")
pairs_df = efiicacy_file.merge(sirna_df, on = "siRNA")
pairs_df = pairs_df.merge(mrna_df, on = "mRNA")


# Construct seq-pairs
pairs_df['seq_pairs'] = pairs_df['siRNA_seq']+'&'+pairs_df['mRNA_seq']


# RNAcofold
for _, row in pairs_df.iterrows():
    pairs = row['seq_pairs']
    id_name = row['siRNA'] + "_" + row['mRNA']
    print(f"Processing {id_name}...")

    proc = subprocess.Popen(
        ['/opt/conda/envs/sirnaenv/bin/RNAcofold', '-p', f"--id-prefix={id_name}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, error = proc.communicate(pairs)

path = "./"
files = os.listdir(path)
    
for file in files:
    if "_0001_dp.ps" not in file or '.bpp' in file:
        continue

    name = file.replace("_0001_dp.ps", "")
    temp = open(path + file).readlines()
    start_flag = False
    os.makedirs(f"RNAcofold_bp_file", exist_ok=True)
    f = open(f"RNAcofold_bp_file/" + file + ".bpp", "w")
        
    for line in temp:
        line = line.strip()

        if "start of base pair probability data" in line:
            start_flag = True

        if start_flag == True and "ubox" in line:
            line = line.strip().split()
            assert(len(line) == 4)
            i, j, prob, _ = line
            prob = float(prob)
            f.write(str(i) + " " + str(j) + " " + str(prob*prob) + "\n")
    f.close()
    
#-------Part 3 (RNAcofold SVD)------------
max_sirna_len = 30
max_mrna_len = 250000
matrix_size = max_sirna_len + max_mrna_len + 2
n_components = 50

file_paths_cofold = glob.glob(f'RNAcofold_bp_file/*dp.ps.bpp')
            
# Use CuPy for GPU operations in this block
with cp.cuda.Device(1): # Specifies GPU device 0. Change to 1 if you need device 1.
    for file_path in file_paths_cofold:
        pos_data = np.loadtxt(file_path, usecols=[0, 1, 2])
        pos_data_gpu = cp.asarray(pos_data) # Convert to CuPy array

        zero_matrix = cp.zeros((matrix_size, matrix_size), dtype=cp.float32) # CuPy zeros
        zero_matrix[pos_data_gpu[:, 0].astype(cp.int32) - 1, pos_data_gpu[:, 1].astype(cp.int32) - 1] = pos_data_gpu[:, 2] # Use cp.int32

        mask = cp.ones(matrix_size, dtype=cp.bool_) # CuPy ones
        mask[max_sirna_len:max_sirna_len + 2] = False
        pos_matrix_gpu = zero_matrix[mask][:, mask] # Perform slicing on GPU
        pos_matrix_gpu = pos_matrix_gpu + pos_matrix_gpu.T - cp.diag(cp.diag(pos_matrix_gpu)) # CuPy operations


        data_matrix_cpu = cp.asnumpy(pos_matrix_gpu) # Convert back to NumPy for CPU SVD
        sparse_matrix = csr_matrix(data_matrix_cpu) # csr_matrix takes NumPy array

        
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        reduced_data = svd.fit_transform(sparse_matrix) # TruncatedSVD works on NumPy

        directory_path_cofold = f'RNAcofold_reduced_matrix'+ str(n_components)
            
        if not os.path.exists(directory_path_cofold):
            os.mkdir(directory_path_cofold)
                
        np.save(f'RNAcofold_reduced_matrix'+ str(n_components) +'/'+file_path.split('/')[-1].split('_dp.ps.bpp')[0]+'.npy',reduced_data)

        print (file_path + " finished!")

#-----------4-----------        
# reduced_matrix
path = f"RNAcofold_reduced_matrix50"
files = os.listdir(path)

df = []
first_parts = []
remaining_parts = []

for file in files:
    file_name = file.replace('_0001.npy', '')
    parts = file_name.split('_', 1)
    first_parts.append(parts[0])
    
    remaining_parts.append(parts[1])
    data = np.load(path + "/" + file)
    data = data.mean(0)
    df.append(data)

df_res = pd.DataFrame(df)

df_res.insert(0, 'siRNA', first_parts)
df_res.insert(1, 'mRNA', remaining_parts)

df_res['index'] = df_res['siRNA'] + '_' + df_res['mRNA']
df_res.set_index('index', inplace=True)
df_res.drop(['siRNA', 'mRNA'], axis=1, inplace=True)
current_path = os.getcwd()
output_path_dir = os.path.join(current_path, f"preprocess")
        
os.makedirs(output_path_dir,exist_ok=True)
output_path_cofold = os.path.join(output_path_dir,f"con_matrix.txt") # Standardized filename
df_res.to_csv(output_path_cofold,header = False)
#print("Saved to:", os.path.abspath("con_matrix_meanSum50.txt"))
current_dir = os.getcwd()

# Loop over all files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".ps"):
        file_path = os.path.join(current_dir, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        

#---------------------mRNA--------------------
# FIX: Changed RNAcofold to RNAfold for single sequence folding
for ind, row in mrna_df.iterrows():
    seq = row[1]
    
    id_name = row[0]

    print (id_name)

    proc = subprocess.Popen(['/opt/conda/envs/sirnaenv/bin/RNAfold','-p',"--id-prefix=" + id_name], # FIX: Changed to RNAfold
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)

    output, error = proc.communicate(seq)
    
path_mrna = "./"
files_mrna = os.listdir(path_mrna)
for file in files_mrna:
    if "_0001_dp.ps" not in file or '.bpp' in file:
        continue
    name = file.replace("_0001_dp.ps", "")
    temp = open(path_mrna + file).readlines()
    start_flag = False
    os.makedirs(f"RNAfold_bp_file_mrna", exist_ok=True)
    f_mrna = open(f"RNAfold_bp_file_mrna/" + file + ".bpp", "w")

    for line in temp:
        line = line.strip()
        if "start of base pair probability data" in line:
            start_flag = True
        if start_flag == True and "ubox" in line:
            line = line.strip().split()
            assert(len(line) == 4)
            i, j, prob, _ = line
            prob = float(prob)
            f_mrna.write(str(i) + " " + str(j) + " " + str(prob*prob) + "\n")
    f_mrna.close()
        
#---------------3 (mRNA SVD)---------

matrix_size_mrna = 250000
n_components = 100

file_paths_mrna = glob.glob(f'RNAfold_bp_file_mrna/*dp.ps.bpp')

# Use CuPy for GPU operations in this block
with cp.cuda.Device(1): # Specifies GPU device 0
    for file_path in file_paths_mrna:
        pos_data = np.loadtxt(file_path, usecols=[0, 1, 2])
        pos_data_gpu = cp.asarray(pos_data) # Convert to CuPy array


        pos_matrix_gpu = cp.zeros((matrix_size_mrna, matrix_size_mrna), dtype=cp.float32) # CuPy zeros
        pos_matrix_gpu[pos_data_gpu[:, 0].astype(cp.int32) - 1, pos_data_gpu[:, 1].astype(cp.int32) - 1] = pos_data_gpu[:, 2] # Use cp.int32
        pos_matrix_gpu = pos_matrix_gpu + pos_matrix_gpu.T - cp.diag(cp.diag(pos_matrix_gpu)) # CuPy operations
        
        data_matrix_cpu = cp.asnumpy(pos_matrix_gpu) # Convert back to NumPy for CPU SVD
        sparse_matrix = csr_matrix(data_matrix_cpu)

            
        svd = TruncatedSVD(n_components=100, random_state=0)
        reduced_data = svd.fit_transform(sparse_matrix) # TruncatedSVD works on NumPy
        directory_path_mrna = f'RNAfold_reduced_matrix_mrna'+ str(n_components)

        if not os.path.exists(directory_path_mrna):
            os.mkdir(directory_path_mrna)

        np.save(f'RNAfold_reduced_matrix_mrna'+ str(n_components) + '/' + file_path.split('/')[-1].split('_dp.ps.bpp')[0]+'.npy',reduced_data)
            
#------------------4---------------------

path_mrna = f"RNAfold_reduced_matrix_mrna100"
files = os.listdir(path_mrna)


df_mrna = []
first_parts_mrna = []

for file in files:
    file_name = file.replace('_0001.npy', '')

    mrna_id = file_name  # Corrected for single ID file names
    first_parts_mrna.append(mrna_id)

    data = np.load(path_mrna + "/" + file)
    data = data.mean(0)
    df_mrna.append(data)

df_res_mrna = pd.DataFrame(df_mrna)

df_res_mrna.index = first_parts_mrna # FIX: Set index correctly

output_path_mrna = os.path.join(output_path_dir,f"self_mRNA_matrix.txt") # Standardized filename
df_res_mrna.to_csv(output_path_mrna, header=False) # FIX: Removed .iloc[[0]]

current_dir = os.getcwd()

# Loop over all files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".ps"):
        file_path = os.path.join(current_dir, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")
#---------------------siRNA-------------------
# FIX: Changed RNAcofold to RNAfold for single sequence folding
for ind, row in sirna_df.iterrows():
    seq = row[1]
    
    id_name = row[0]
    
    proc = subprocess.Popen(['/opt/conda/envs/sirnaenv/bin/RNAfold','-p',"--id-prefix=" + id_name], # FIX: Changed to RNAfold
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text = True)

    output, error = proc.communicate(seq)
    
#-------------2------------
path_sirna = "./"
files_sirna = os.listdir(path_sirna)
for file in files_sirna:
    if "_0001_dp.ps" not in file or '.bpp' in file:
        continue
    name = file.replace("_0001_dp.ps", "")
    temp = open(path_sirna + file).readlines()
    start_flag = False
    os.makedirs(f"RNAfold_bp_file_sirna", exist_ok=True)
    f_sirna = open(f"RNAfold_bp_file_sirna/" + file + ".bpp", "w")
    for line in temp:
        line = line.strip()
        if "start of base pair probability data" in line:
            start_flag = True
        if start_flag == True and "ubox" in line:
            line = line.strip().split()
            assert(len(line) == 4)
            i, j, prob, _ = line
            prob = float(prob)
            f_sirna.write(str(i) + " " + str(j) + " " + str(prob*prob) + "\n")
    f_sirna.close()
        
#-------------3 (siRNA SVD)-----------------
matrix_size_sirna = 30
n_components_sirna = 6  

file_paths_sirna = glob.glob(f'RNAfold_bp_file_sirna/*dp.ps.bpp')

# Use CuPy for GPU operations in this block
with cp.cuda.Device(1): # Specifies GPU device 0
    for file_path in file_paths_sirna:
        # Retained original try-except for SVD calculation as per your provided code.
        try:
            pos_data = np.loadtxt(file_path, usecols=[0, 1, 2])
            pos_data_gpu = cp.asarray(pos_data) # Convert to CuPy array
            
            if len(pos_data) < n_components_sirna:
                reduced_data = cp.zeros((matrix_size_sirna, 6), dtype=cp.float32).get() # CuPy zeros, then .get() to NumPy
            else:
                pos_matrix_gpu = cp.zeros((matrix_size_sirna, matrix_size_sirna), dtype=cp.float32) # CuPy zeros
                pos_matrix_gpu[pos_data_gpu[:, 0].astype(cp.int32) - 1, pos_data_gpu[:, 1].astype(cp.int32) - 1] = pos_data_gpu[:, 2] # Use cp.int32
                pos_matrix_gpu = pos_matrix_gpu + pos_matrix_gpu.T - cp.diag(cp.diag(pos_matrix_gpu)) # CuPy operations

                data_matrix_cpu = cp.asnumpy(pos_matrix_gpu) # Convert back to NumPy for CPU SVD
                sparse_matrix = csr_matrix(data_matrix_cpu)

                svd = TruncatedSVD(n_components=6, random_state=0)
                reduced_data = svd.fit_transform(sparse_matrix) # TruncatedSVD works on NumPy
        except: # Original broad except block
            reduced_data = np.zeros((matrix_size_sirna, n_components_sirna), dtype=np.float32) # NumPy zeros fallback
        
        directory_path_sirna = f'RNAfold_reduced_matrix_sirna'+ str(n_components_sirna)

        if not os.path.exists(directory_path_sirna):
            os.mkdir(directory_path_sirna)

        np.save(f'RNAfold_reduced_matrix_sirna'+ str(n_components_sirna) + '/' + file_path.split('/')[-1].split('_dp.ps.bpp')[0]+'.npy',reduced_data)
            
#------------------4--------------
path_sirna = f"RNAfold_reduced_matrix_sirna6"
files_sirna = os.listdir(path_sirna)

df_sirna = []
first_parts_sirna = []

for file in files_sirna:
    file_name = file.replace('_0001.npy', '')
    sirna_id = file_name # Corrected for single ID file names
    first_parts_sirna.append(sirna_id)

    data = np.load(os.path.join(path_sirna, file))
    result_data = data.mean(0)

    df_sirna.append(result_data)


# create the DataFrame
df_res_sirna = pd.DataFrame(df_sirna)
df_res_sirna.index = first_parts_sirna # FIX: Set index correctly

output_path_sirna = os.path.join(output_path_dir,f"self_siRNA_matrix.txt") # Standardized filename
df_res_sirna.to_csv(output_path_sirna, header=False) # FIX: Removed .iloc[[0]]

current_dir = os.getcwd()

# Loop over all files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".ps"):
        file_path = os.path.join(current_dir, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

#---------------------match position------------------------------

import pandas as pd
from Bio import SeqIO

# --- Load base data ---
eff_df = pd.read_csv("full_efficacy.csv")

# --- Load siRNA sequences ---
sirna_map = {
    rec.id: str(rec.seq).upper().replace("T", "U")
    for rec in SeqIO.parse("full_siRNA.fas", "fasta")
}

# --- Load mRNA sequences ---
mrna_map = {
    rec.id: str(rec.seq).upper().replace("T", "U")
    for rec in SeqIO.parse("full_mRNA.fas", "fasta")
}

# --- Add sequences ---
eff_df['siRNA_seq'] = eff_df['siRNA'].map(sirna_map)
eff_df['mRNA_seq'] = eff_df['mRNA'].map(mrna_map)  
eff_df['mRNA_seq_RNA-FM'] = eff_df['mRNA_seq'].str.replace("T", "U")


# Drop rows with missing mappings
eff_df = eff_df.dropna()

# --- Compute match position ---
c = 0 
def reverse_complement_rna(seq):
    return seq.strip().upper().translate(str.maketrans("AUGC", "UACG"))[::-1]

def find_match_pos(row):
    global c 
    try: 
        sirna = reverse_complement_rna(row['siRNA_seq'])
        mrna = row['mRNA_seq_RNA-FM'].strip().upper()
        return mrna.find(sirna)
    except Exception as e:
        c+=1
        print("Exception")
        return -1

eff_df['pos'] = eff_df.apply(find_match_pos, axis=1)
eff_df = eff_df[eff_df['pos'] != -1]  # optional: filter invalid matches

print("c=",c)

# --- Save final merged dataset ---
final_cols = ['siRNA', 'mRNA', 'efficacy', 'siRNA_seq', 'mRNA_seq', 'mRNA_seq_RNA-FM', 'pos']
eff_df[final_cols].to_csv("all_data.csv", index=False)

print(f"✅ Final merged dataset saved: {len(eff_df)} rows → viral_all_data.csv")
