import numpy
import pandas
import math
import torch
from transformers import AutoTokenizer, AutoModel

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

try:
    MP_RNA_MODEL_NAME = "yangheng/MP-RNA"
    GLOBAL_MP_RNA_TOKENIZER = AutoTokenizer.from_pretrained(MP_RNA_MODEL_NAME)
    GLOBAL_MP_RNA_MODEL = AutoModel.from_pretrained(MP_RNA_MODEL_NAME)
    GLOBAL_MP_RNA_MODEL.eval()
except Exception as e:
    GLOBAL_MP_RNA_TOKENIZER = None
    GLOBAL_MP_RNA_MODEL = None

MP_RNA_EMBEDDING_DIM = 768

def get_mp_rna_sequence_embedding(seq: str) -> numpy.ndarray:
    if GLOBAL_MP_RNA_MODEL is None or GLOBAL_MP_RNA_TOKENIZER is None:
        return numpy.zeros(MP_RNA_EMBEDDING_DIM, dtype=numpy.float32)

    inputs = GLOBAL_MP_RNA_TOKENIZER(seq, return_tensors="pt", padding=True, truncation=True)

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

def cal_thermo_feature(sequence, intermolecular_initiation=4.09, symmetry_correction=0.43):
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
    single_sum.append(round(sum_stability, 2))

    return single_sum

def countGC(seq):
    seq = seq.upper().replace("T", "U")
    gc_percent = (seq.count("G") + seq.count("C")) / len(seq)
    return round(gc_percent, 3)

def combined_kmer_freq(seq):
    seq = seq.upper().replace("T", "U")
    single = {base: seq.count(base) / len(seq) for base in "ACGU"}
    all_double_base = [a + b for a in "ACGU" for b in "ACGU"]
    double_freq = {dbase: 0 for dbase in all_double_base}
    total_double = len(seq) - 1
    for i in range(total_double):
        double_base = seq[i:i+2]
        double_freq[double_base] += 1
    double_freq = {k: v / total_double for k, v in double_freq.items()}
    return {**single, **double_freq}

def rules_scores(seq):
    seq = seq.upper().replace("T", "U")
    scores = []
    for i in range(19):
        score = position_scores[i][seq[i]]
        scores.append(score)
    return scores