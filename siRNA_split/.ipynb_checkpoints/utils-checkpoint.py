import numpy
import pandas
import math
import torch
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()
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
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 0, "C": -1, "G": 0, "U": 0},
    {"A": 1, "C": -1, "G": 0, "U": -1},
    {"A": 0, "C": 0, "G": 1, "U": 1},
    {"A": 1, "C": 0, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1}
]

# try:
#     MP_RNA_MODEL_NAME = "yangheng/MP-RNA"
#     GLOBAL_MP_RNA_TOKENIZER = AutoTokenizer.from_pretrained(MP_RNA_MODEL_NAME)
#     GLOBAL_MP_RNA_MODEL = AutoModel.from_pretrained(MP_RNA_MODEL_NAME)
#     GLOBAL_MP_RNA_MODEL.eval()
# except Exception as e:
#     GLOBAL_MP_RNA_TOKENIZER = None
#     GLOBAL_MP_RNA_MODEL = None

MP_RNA_EMBEDDING_DIM = 768

def get_mp_rna_sequence_embedding(seq: str) -> numpy.ndarray:
    MP_RNA_MODEL_NAME = "yangheng/MP-RNA"
    GLOBAL_MP_RNA_TOKENIZER = AutoTokenizer.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True)
    GLOBAL_MP_RNA_MODEL = AutoModel.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True)
    GLOBAL_MP_RNA_MODEL.eval()
    # if GLOBAL_MP_RNA_MODEL is None or GLOBAL_MP_RNA_TOKENIZER is None:
    #     return numpy.zeros(MP_RNA_EMBEDDING_DIM, dtype=numpy.float32)

    inputs = GLOBAL_MP_RNA_TOKENIZER(seq, return_tensors="pt")

    with torch.no_grad():
        outputs = GLOBAL_MP_RNA_MODEL(**inputs)

    sequence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    return sequence_embedding

def get_pos_embedding(index, d_model, t):
    if index % 2 == 0:
        return math.sin(t / pow(10000, index / (d_model - 1)))
    else:
        return math.cos(t / pow(10000, index / (d_model - 1)))

def get_pos_embedding_sequence(mrna_start_pos, len_sirna, d_model):
    pe_list = []
    for offset in range(len_sirna):
        t = mrna_start_pos + offset
        for i in range(d_model):
            pe = get_pos_embedding(i, d_model, t)
            pe_list.append(pe)
    return pe_list

def build_kmers(sequence):
    kmers = []
    n_kmers = len(sequence) - 1

    for i in range(n_kmers):
        kmer = sequence[i:i + 2]
        kmers.append(kmer)
    return kmers

def cal_thermo_feature(sequence,intermolecular_initiation=4.09, symmetry_correction=0.43):
    sum_stability = 0
    single_sum = []

    bimers = build_kmers(sequence)

    if sequence[0] == 'A':
        sum_stability += 0.45
    if sequence[18] == 'U': 
        sum_stability += 0.45

    bimer_values = {
        'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33,
        'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11,
        'GU': -2.24, 'AC': -2.24, 'GA': -2.35, 'UC': -2.35,
        'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42
    }
    for b in bimers:
        stability_value = bimer_values.get(b, 0)
        single_sum.append(stability_value)
        sum_stability += stability_value

    sum_stability += intermolecular_initiation
    sum_stability += symmetry_correction
    single_sum.append(round(sum_stability,2))

    return single_sum

def countGC(seq):
    seq = seq.upper().replace("T","U")
    gc_percent = (seq.count("G")+seq.count("C"))/len(seq)
    return round(gc_percent,3)

def single_freq(seq):
    seq = seq.upper().replace("T","U")
    freq = {base: seq.count(base) / len(seq) for base in "ACGU"}

    return freq

def double_freq(seq):
    seq = seq.upper().replace("T","U")
    all_double_base = [a + b for a in "ACGU" for b in "ACGU"]

    double_base_freq = {dbase: 0 for dbase in all_double_base}
    totabl_base = len(seq) - 1 

    for i in range(totabl_base):
        double_base = seq[i:i+2]
        double_base_freq[double_base] += 1 

    double_base_freq = {key : value / totabl_base for key, value in double_base_freq.items()}

    return double_base_freq

def triple_freq(seq):
    seq = seq.upper().replace("T","U")

    all_triple_base = [a + b + c for a in "ACGU" for b in "ACGU" for c in "ACGU"]

    triple_base_freq = {tbase: 0 for tbase in all_triple_base} 
    totabl_base = len(seq) - 2 

    for i in range(totabl_base):
        triple_base = seq[i:i+3]
        triple_base_freq[triple_base] += 1 

    triple_base_freq = {key : value / totabl_base for key, value in triple_base_freq.items()}

    return triple_base_freq

def quadruple_freq(seq):
    seq = seq.upper().replace("T","U")

    all_quadruple_base = [a + b + c + d for a in "ACGU" for b in "ACGU" for c in "ACGU" for d in "ACGU"]

    quadruple_base_freq = {tbase: 0 for tbase in all_quadruple_base} 
    totabl_base = len(seq) - 3 

    for i in range(totabl_base):
        quadruple_base = seq[i:i+4]
        quadruple_base_freq[quadruple_base] += 1 

    quadruple_base_freq = {key : value / totabl_base for key, value in quadruple_base_freq.items()}

    return quadruple_base_freq

def quintuple_freq(seq):
    seq = seq.upper().replace("T","U")

    all_quintuple_base = [a + b + c + d + e for a in "ACGU" for b in "ACGU" for c in "ACGU" for d in "ACGU" for e in "ACGU"]

    quintuple_base_freq = {tbase: 0 for tbase in all_quintuple_base} 
    totabl_base = len(seq) - 4 

    for i in range(totabl_base):
        quintuple_base = seq[i:i+5]
        quintuple_base_freq[quintuple_base] += 1 

    quintuple_base_freq = {key : value / totabl_base for key, value in quintuple_base_freq.items()}

    return quintuple_base_freq

def one_hot_encode(score):
    if score == -1:
        return [1, 0, 0]
    elif score == 0:
        return [0, 1, 0]
    elif score == 1:
        return [0, 0, 1]

def rules_scores(seq):
    seq = seq.upper().replace("T","U")
    one_hot_scores = []
    for i in range(19):
        score = position_scores[i][seq[i]]
        one_hot_scores.extend(one_hot_encode(score))
    return one_hot_scores
