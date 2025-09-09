import numpy
import math
import itertools # Needed for k-mer generation
import torch
from transformers import AutoTokenizer, AutoModel, logging

# Suppress warnings from transformers library to reduce console clutter
# logging.set_verbosity_error()

# # --- Global MP-RNA Transformer Model and Tokenizer Initialization ---
# # These are loaded once to avoid repeated loading during feature extraction.
# MP_RNA_MODEL_NAME = "yangheng/MP-RNA"
# MP_RNA_EMBEDDING_DIM = 768 # Default embedding dimension from the MP-RNA model

# # Initialize global variables for tokenizer and model
# GLOBAL_MP_RNA_TOKENIZER = None
# GLOBAL_MP_RNA_MODEL = None
# GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# try:
#     print(f"BugUtils: Attempting to load MP-RNA Transformer tokenizer and model on {GLOBAL_DEVICE}...")
#     GLOBAL_MP_RNA_TOKENIZER = AutoTokenizer.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True)
#     GLOBAL_MP_RNA_MODEL = AutoModel.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True).to(GLOBAL_DEVICE)
#     GLOBAL_MP_RNA_MODEL.eval() # Set model to evaluation mode for inference
#     print("BugUtils: MP-RNA Transformer loaded successfully.")
# except Exception as e:
#     print(f"BugUtils: ERROR: Could not load MP-RNA Transformer model '{MP_RNA_MODEL_NAME}'.")
#     print(f"BugUtils: Please ensure you have internet access for the first download or the model is cached.")
#     print(f"BugUtils: Error details: {e}")
#     print(f"BugUtils: MP-RNA Transformer will NOT be available for embedding generation. Sequence embeddings will be zeros (dimension: {MP_RNA_EMBEDDING_DIM}).")
    # GLOBAL_MP_RNA_TOKENIZER and GLOBAL_MP_RNA_MODEL remain None, get_mp_rna_sequence_embedding handles this.
# --- End Global Model Initialization ---


# --- siRNA position scores (Highly relevant for siRNA efficacy) ---
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

# --- General Sequence Utility Functions ---

# def obtain_one_hot_feature_for_one_sequence_1(seq1, max_len):
#     """
#     Converts a sequence to one-hot encoding, padding or truncating to max_len.
#     Handles 'N' as all zeros.
#     """
#     # Mapping for 'ACGTN'
#     mapping = dict(zip("NACGT", range(5)))
    
#     seq_numeric = [mapping.get(i.upper(), 0) for i in seq1]

#     if len(seq_numeric) > max_len:
#         seq_numeric = seq_numeric[:max_len]
    
#     padding_len = max_len - len(seq_numeric)
    
#     # Unit array for one-hot encoding, handling 'N' (mapped to 0) as all zeros
#     # 0 -> [0,0,0,0], 1->[1,0,0,0], 2->[0,1,0,0] etc.
#     unit_arr = numpy.concatenate((numpy.zeros((1, 4), dtype=numpy.uint8), numpy.eye(4, dtype=numpy.uint8)))

#     encoded_seq = unit_arr[seq_numeric]
    
#     zero_arr = numpy.zeros((padding_len, 4), dtype=numpy.uint8)
    
#     return numpy.concatenate((encoded_seq, zero_arr)).flatten()


# def get_mp_rna_sequence_embedding(seq: str, target_dim: int = MP_RNA_EMBEDDING_DIM) -> numpy.ndarray:
#     """
#     Generates a sequence embedding using the pre-trained MP-RNA model.
#     Model and tokenizer are loaded globally to ensure efficiency.
#     Returns zero vector if model not loaded or sequence is invalid/empty.
#     """
#     # If model/tokenizer failed to load globally, or if sequence is empty/invalid, return zeros
#     # if GLOBAL_MP_RNA_MODEL is None or GLOBAL_MP_RNA_TOKENIZER is None or not isinstance(seq, str) or not seq.strip():
#         # Pad with zeros to target_dim if it's different from MP_RNA_EMBEDDING_DIM
#         # if target_dim != MP_RNA_EMBEDDING_DIM:
#         #      # print(f"BugUtils Warning: MP-RNA model not available or empty sequence. Returning {target_dim}-dim zeros.")
#         # return numpy.zeros(target_dim, dtype=numpy.float32)

#     # Tokenize the sequence
#     # max_length for MP-RNA depends on the model variant. 128 is a safe general value.
#     # Adjust max_length based on the expected length of your UTR/CDS sequences.
#     # RNA-FM sequences often have 'U' so ensure tokenizer handles 'U' if not converted to 'T'
#     inputs = GLOBAL_MP_RNA_TOKENIZER(seq, return_tensors="pt", truncation=True, padding='max_length', max_length=128) # Max length is context dependent

#     # Ensure inputs are on the same device as the model (CPU or CUDA)
#     inputs = {k: v.to(GLOBAL_DEVICE) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = GLOBAL_MP_RNA_MODEL(**inputs)

#     # Extract [CLS] token embedding, move to CPU, and convert to numpy
#     # The [CLS] token is typically the first token (index 0) in the last hidden state.
#     sequence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

#     # If target_dim is specified and differs from MP_RNA_EMBEDDING_DIM (256), handle it.
#     if sequence_embedding.shape[0] != target_dim:
#         if sequence_embedding.shape[0] < target_dim:
#             # Pad with zeros if the transformer output is smaller than target_dim
#             padded_embedding = numpy.pad(sequence_embedding, (0, target_dim - sequence_embedding.shape[0]), 'constant')
#             sequence_embedding = padded_embedding
#         else:
#             # Truncate if the transformer output is larger than target_dim
#             sequence_embedding = sequence_embedding[:target_dim]
#         # print(f"BugUtils Warning: MP-RNA embedding dimension ({MP_RNA_EMBEDDING_DIM}) did not match requested target_dim ({target_dim}). Padded/truncated embedding.")

#     return sequence_embedding


# --- Positional Encoding (Specific to custom Transformer; d_model should match embedding_dim) ---
def get_pos_embedding(index, d_model, t):
    """Calculates a single positional embedding value."""
    # Ensure d_model is at least 1 to avoid ZeroDivisionError
    denominator = pow(10000, index / (d_model - 1)) if d_model > 1 else 1
    if index % 2 == 0:
        return math.sin(t / denominator)
    else:
        return math.cos(t / denominator)

def get_pos_embedding_sequence(mrna_start_pos, len_seq, max_seq_len, d_model):
    """
    Generates positional embeddings for a sequence, padding to max_seq_len.
    The d_model here should match the embedding_dim used in the TransformerEncoder.
    """
    pe_list = []
    for offset in range(len_seq):
        t = mrna_start_pos + offset
        for i in range(d_model):
            pe = get_pos_embedding(i, d_model, t)
            pe = round(pe, 4)
            pe_list.append(pe)
    
    expected_padded_len = max_seq_len * d_model
    
    if len(pe_list) < expected_padded_len:
        pe_list.extend([0.0] * (expected_padded_len - len(pe_list)))
    elif len(pe_list) > expected_padded_len:
        pe_list = pe_list[:expected_padded_len]
        
    return pe_list


# --- Thermodynamics (Highly relevant for siRNA efficacy) ---
def build_kmers(sequence, k=2): # Added k parameter with default
    """Builds k-mers (dinucleotides by default) from a sequence."""
    kmers = []
    if len(sequence) >= k:
        n_kmers = len(sequence) - k + 1
        for i in range(n_kmers):
            kmer = sequence[i:i + k]
            kmers.append(kmer)
    return kmers

def cal_thermo_feature(sequence, max_len_expected_output, intermolecular_initiation=4.09, symmetry_correction=0.43): # Renamed max_sirna_len for clarity
    """
    Calculates thermodynamic features for an siRNA sequence, padding to a fixed length.
    This function is highly relevant for siRNA efficacy prediction.
    """
    seq = sequence.upper().replace("T", "U") # Convert T to U for RNA consistency
    
    sum_stability = 0
    single_sum_values = []

    # siRNA 5' A and 3' U terminal rules (highly specific to siRNA)
    if len(seq) > 0 and seq[0] == 'A':
        sum_stability += 0.45
    if len(seq) >= 19 and seq[18] == 'U': # This line is specific to 19-mer siRNA sequences
        sum_stability += 0.45

    bimers = build_kmers(seq, k=2) # Ensure 2-mers are built for the bimer_values_dict

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

    # Intermolecular initiation and symmetry correction (highly specific to siRNA duplex formation)
    sum_stability += intermolecular_initiation
    sum_stability += symmetry_correction
    
    single_sum_values.append(round(sum_stability, 2))

    # Pad/truncate to the expected output length
    if len(single_sum_values) < max_len_expected_output:
        single_sum_values.extend([0.0] * (max_len_expected_output - len(single_sum_values)))
    elif len(single_sum_values) > max_len_expected_output:
        single_sum_values = single_sum_values[:max_len_expected_output]
        
    return single_sum_values

# --- GC Percentage (Robust handling of empty sequences) ---
def countGC(seq):
    """
    Calculates GC percentage for a sequence. Returns 0.0 for empty sequences.
    """
    seq = seq.upper()
    if not seq: # Handle empty sequence safely
        return 0.0
    gc_count = (seq.count("G") + seq.count("C"))
    gc_percent = gc_count / len(seq)
    return round(gc_percent, 3)


# --- K-mer functions (Robust handling of short sequences) ---
def get_kmer_freq(seq, k):
    """
    Calculates k-mer frequencies. Returns a fixed-size list of frequencies for all possible k-mers.
    Handles sequences too short to form k-mers by returning all zeros.
    """
    seq = seq.upper().replace("T","U") # Convert T to U for RNA consistency
    
    bases = "ACGU"
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    
    kmer_freq = {kmer: 0.0 for kmer in all_kmers}

    total_kmers = len(seq) - k + 1
    
    if total_kmers <= 0: # If sequence is too short to form k-mers, return all zeros
        return list(kmer_freq.values())
        
    for i in range(total_kmers):
        kmer = seq[i:i+k]
        if kmer in kmer_freq:
            kmer_freq[kmer] += 1
    
    # Normalize frequencies
    for key in kmer_freq:
        kmer_freq[key] /= total_kmers
            
    return list(kmer_freq.values())

# Wrappers for specific k-mer lengths
def single_freq(seq, *args):
    """Wrapper for 1-mer frequencies."""
    return get_kmer_freq(seq, 1)

def double_freq(seq, *args):
    """Wrapper for 2-mer frequencies."""
    return get_kmer_freq(seq, 2)

def triple_freq(seq, *args):
    """Wrapper for 3-mer frequencies."""
    return get_kmer_freq(seq, 3)

def quadruple_freq(seq, *args):
    """Wrapper for 4-mer frequencies."""
    return get_kmer_freq(seq, 4)

def quintuple_freq(seq, *args):
    """Wrapper for 5-mer frequencies."""
    return get_kmer_freq(seq, 5)


# --- siRNA Rules Scores (Highly relevant for siRNA efficacy) ---
def one_hot_encode(score):
    """One-hot encode the score (-1, 0, 1)."""
    if score == -1:
        return [1, 0, 0]
    elif score == 0:
        return [0, 1, 0]
    elif score == 1:
        return [0, 0, 1]
    return [0, 0, 0] # For any other score, or invalid

def rules_scores(seq, max_rules_length=19):
    """
    Obtain siRNA each position scores with one-hot encoding, padded to a fixed length.
    This function is highly specific and relevant for siRNA efficacy prediction.
    """
    seq = seq.upper().replace("T","U") # Convert T to U for RNA consistency
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