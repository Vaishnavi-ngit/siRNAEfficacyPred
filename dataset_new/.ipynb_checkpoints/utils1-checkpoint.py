import numpy
import math
import itertools
import torch
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()

MP_RNA_MODEL_NAME = "yangheng/MP-RNA"

GLOBAL_MP_RNA_TOKENIZER = None
GLOBAL_MP_RNA_MODEL = None

GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    GLOBAL_MP_RNA_TOKENIZER = AutoTokenizer.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True)
    GLOBAL_MP_RNA_MODEL = AutoModel.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True).to(GLOBAL_DEVICE)
    GLOBAL_MP_RNA_MODEL.eval()
except:
    GLOBAL_MP_RNA_TOKENIZER = None
    GLOBAL_MP_RNA_MODEL = None

position_scores = [
    {"A": -1, "C": 1, "G": 1, "U": -1},
    {"A": -1, "C": 0, "G": 1, "U": -1},
    {"A": 1, "C": -1, "G": 1, "U": -1},
    {"A": 0, "C": -1, "G": 0, "U": 1},
    {"A": 1, "C": 0, "G": 0, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": 1, "U": -1},
    {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 0, "C": 0, "G": -1, "U": -1},
    {"A": 1, "C": 1, "G": 1, "U": 1},
    {"A": 0, "C": 1, "G": 1, "U": 0},
    {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 1, "C": -1, 'G': -1, "U": 1},
    {"A": 0, "C": -1, "G": 0, "U": 0},
    {"A": 1, "C": -1, "G": 0, "U": -1},
    {"A": 0, "C": 0, "G": 1, "U": 1},
    {"A": 1, "C": 0, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1}
]

MP_RNA_EMBEDDING_DIM = 768

def get_mp_rna_sequence_embedding(seq: str) -> numpy.ndarray:
    inputs = GLOBAL_MP_RNA_TOKENIZER(seq, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(GLOBAL_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = GLOBAL_MP_RNA_MODEL(**inputs)

    sequence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    return sequence_embedding

def get_pos_embedding(index, d_model, t):
    denominator = pow(10000, index / (d_model - 1)) if d_model > 1 else 1
    if index % 2 == 0:
        return math.sin(t / denominator)
    else:
        return math.cos(t / denominator)

def get_pos_embedding_sequence(mrna_start_pos, len_sirna, max_sirna_len, d_model):
    pe_list = []
    for offset in range(len_sirna):
        t = mrna_start_pos + offset
        for i in range(d_model):
            pe = get_pos_embedding(i, d_model, t)
            pe = round(pe, 4)
            pe_list.append(pe)
    
    expected_padded_len = max_sirna_len * d_model
    
    if len(pe_list) < expected_padded_len:
        pe_list.extend([0.0] * (expected_padded_len - len(pe_list)))
    elif len(pe_list) > expected_padded_len:
        pe_list = pe_list[:expected_padded_len]
        
    return pe_list

def build_kmers(sequence):
    kmers = []
    if len(sequence) >= 2:
        n_kmers = len(sequence) - 1
        for i in range(n_kmers):
            kmer = sequence[i:i + 2]
            kmers.append(kmer)
    return kmers

def cal_thermo_feature(sequence, max_sirna_len, intermolecular_initiation=4.09, symmetry_correction=0.43):
    seq = sequence.upper().replace("T", "U")
    
    sum_stability = 0
    single_sum_values = []

    if len(seq) > 0 and seq[0] == 'A':
        sum_stability += 0.45
    if len(seq) >= 19 and seq[18] == 'U':
        sum_stability += 0.45

    bimers = build_kmers(seq)

    bimer_values_dict = {
        'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33,
        'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11,
        'GU': -2.24, 'AC': -2.24, 'GA': -2.35, 'UC': -2.35,
        'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42
    }

    for b in bimers:
        stability_value = bimer_values_dict.get(b, 0)
        single_sum_values.append(stability_value)
        sum_stability += stability_value

    sum_stability += intermolecular_initiation
    sum_stability += symmetry_correction
    
    single_sum_values.append(round(sum_stability, 2))

    expected_output_len = max_sirna_len 
    
    if len(single_sum_values) < expected_output_len:
        single_sum_values.extend([0.0] * (expected_output_len - len(single_sum_values)))
    elif len(single_sum_values) > expected_output_len:
        single_sum_values = single_sum_values[:expected_output_len]
        
    return single_sum_values

def countGC(seq):
    seq = seq.upper()
    gc_count = (seq.count("G") + seq.count("C"))
    gc_percent = gc_count / len(seq)
    return round(gc_percent, 3)

def get_kmer_freq(seq, k):
    seq = seq.upper().replace("T","U")
    
    bases = "ACGU"
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    
    kmer_freq = {kmer: 0.0 for kmer in all_kmers}

    total_kmers = len(seq) - k + 1
    for i in range(total_kmers):
        kmer = seq[i:i+k]
        if kmer in kmer_freq:
            kmer_freq[kmer] += 1
    
    for key in kmer_freq:
        kmer_freq[key] /= total_kmers
        
    return list(kmer_freq.values())

def single_freq(seq, *args):
    return get_kmer_freq(seq, 1)

def double_freq(seq, *args):
    return get_kmer_freq(seq, 2)

def triple_freq(seq, *args):
    return get_kmer_freq(seq, 3)

def quadruple_freq(seq, *args):
    return get_kmer_freq(seq, 4)

def quintuple_freq(seq, *args):
    return get_kmer_freq(seq, 5)

def one_hot_encode(score):
    if score == -1:
        return [1, 0, 0]
    elif score == 0:
        return [0, 1, 0]
    elif score == 1:
        return [0, 0, 1]
    return [0, 0, 0]

def rules_scores(seq, max_rules_length=19):
    seq = seq.upper().replace("T","U")
    one_hot_scores = []
    
    for i in range(max_rules_length):
        if i < len(seq) and i < len(position_scores):
            score = position_scores[i].get(seq[i], 0)
            one_hot_scores.extend(one_hot_encode(score))
        else:
            one_hot_scores.extend([0, 0, 0])
    
    expected_len = max_rules_length * 3
    if len(one_hot_scores) < expected_len:
        one_hot_scores.extend([0] * (expected_len - len(one_hot_scores)))
    elif len(one_hot_scores) > expected_len:
        one_hot_scores = one_hot_scores[:expected_len]

    return one_hot_scores