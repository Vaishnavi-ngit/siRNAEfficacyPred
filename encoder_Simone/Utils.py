import numpy
import math
import itertools
import torch
from transformers import AutoTokenizer, AutoModel, logging

# --- Configuration & Global Setup ---

# Suppress informational warnings from the transformers library
logging.set_verbosity_error()

# Define model constants
MP_RNA_MODEL_NAME = "yangheng/MP-RNA"
# Correct embedding dimension for the yangheng/MP-RNA model is 768
MP_RNA_EMBEDDING_DIM = 768

# Initialize global variables for the model, tokenizer, and device
# This ensures they are loaded only once for efficiency.
GLOBAL_MP_RNA_TOKENIZER = None
GLOBAL_MP_RNA_MODEL = None
GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Loading ---

try:
    # Attempt to load the tokenizer and model from Hugging Face or cache
    GLOBAL_MP_RNA_TOKENIZER = AutoTokenizer.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True)
    GLOBAL_MP_RNA_MODEL = AutoModel.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True).to(GLOBAL_DEVICE)
    GLOBAL_MP_RNA_MODEL.eval() # Set model to evaluation mode for consistent inference
except Exception as e:
    # If loading fails, the model and tokenizer will remain None.
    # The get_mp_rna_sequence_embedding function will handle this case gracefully.
    print(f"Warning: Could not load MP-RNA Transformer model '{MP_RNA_MODEL_NAME}'. "
          f"Sequence embeddings will be zero vectors. Error: {e}")


# --- siRNA Positional Scores ---
# These scores are based on empirical rules for siRNA efficacy.
position_scores = [
    {"A": -1, "C": 1, "G": 1, "U": -1}, {"A": -1, "C": 0, "G": 1, "U": -1},
    {"A": 1, "C": -1, "G": 1, "U": -1}, {"A": 0, "C": -1, "G": 0, "U": 1},
    {"A": 1, "C": 0, "G": 0, "U": 1}, {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": 1, "U": -1}, {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 0, "C": 0, "G": -1, "U": -1}, {"A": 1, "C": 1, "G": 1, "U": 1},
    {"A": 0, "C": 1, "G": 1, "U": 0}, {"A": 1, "C": 0, "G": -1, "U": 0},
    {"A": 1, "C": -1, 'G': -1, "U": 1}, {"A": 0, "C": -1, "G": 0, "U": 0},
    {"A": 1, "C": -1, "G": 0, "U": -1}, {"A": 0, "C": 0, "G": 1, "U": 1},
    {"A": 1, "C": 0, "G": -1, "U": 1}, {"A": 1, "C": -1, "G": -1, "U": 1},
    {"A": 1, "C": -1, "G": -1, "U": 1}
]


# --- Feature Engineering Functions ---

def obtain_one_hot_feature_for_one_sequence_1(seq: str, max_len: int) -> numpy.ndarray:
    """
    Converts a sequence to one-hot encoding, padding or truncating to max_len.
    'N' is handled as an all-zero vector.
    """
    mapping = dict(zip("NACGT", range(5)))
    seq_numeric = [mapping.get(i.upper(), 0) for i in seq]

    # Truncate if sequence is longer than max_len
    if len(seq_numeric) > max_len:
        seq_numeric = seq_numeric[:max_len]

    # Define one-hot mapping: 0 -> [0,0,0,0], 1 -> [1,0,0,0] for A, etc.
    unit_arr = numpy.concatenate((numpy.zeros((1, 4), dtype=numpy.uint8), numpy.eye(4, dtype=numpy.uint8)))
    encoded_seq = unit_arr[seq_numeric]

    # Pad with zeros if sequence is shorter than max_len
    padding_len = max_len - len(encoded_seq)
    if padding_len > 0:
        padding_arr = numpy.zeros((padding_len, 4), dtype=numpy.uint8)
        encoded_seq = numpy.concatenate((encoded_seq, padding_arr))

    return encoded_seq.flatten()


def get_mp_rna_sequence_embedding(seq: str) -> numpy.ndarray:
    """
    Generates a sequence embedding using the pre-trained MP-RNA model.
    Returns a zero vector if the model isn't loaded or the sequence is invalid.
    This version is simplified to always return the model's native embedding size.
    """
    # --- Input Validation ---
    if not isinstance(seq, str) or not seq.strip():
        return numpy.zeros(MP_RNA_EMBEDDING_DIM, dtype=numpy.float32)

    # --- Model Availability Check ---
    if GLOBAL_MP_RNA_MODEL is None or GLOBAL_MP_RNA_TOKENIZER is None:
        return numpy.zeros(MP_RNA_EMBEDDING_DIM, dtype=numpy.float32)

    # --- Tokenization and Inference ---
    try:
        # Tokenize the sequence. 128 is a safe max_length for typical UTR/CDS contexts.
        inputs = GLOBAL_MP_RNA_TOKENIZER(
            seq, return_tensors="pt", truncation=True, padding='max_length', max_length=128
        )
        inputs = {k: v.to(GLOBAL_DEVICE) for k, v in inputs.items()}

        # Perform inference without calculating gradients
        with torch.no_grad():
            outputs = GLOBAL_MP_RNA_MODEL(**inputs)

        # Extract the [CLS] token embedding (represents the whole sequence)
        # and move it to the CPU as a numpy array.
        sequence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        # Ensure the output has the expected dimension, otherwise return zeros as a fallback.
        if sequence_embedding.shape[0] != MP_RNA_EMBEDDING_DIM:
            return numpy.zeros(MP_RNA_EMBEDDING_DIM, dtype=numpy.float32)

        return sequence_embedding

    except Exception as e:
        # *** MODIFICATION FOR DEBUGGING ***
        # Print the error to diagnose the problem
        print(f"Error processing sequence '{seq[:30]}...': {e}")
        # Return a zero vector so the program doesn't crash
        return numpy.zeros(MP_RNA_EMBEDDING_DIM, dtype=numpy.float32)


def get_pos_embedding(index, d_model, t):
    """Calculates a single positional embedding value."""
    denominator = pow(10000, index / (d_model - 1)) if d_model > 1 else 1
    if index % 2 == 0:
        return math.sin(t / denominator)
    return math.cos(t / denominator)


def get_pos_embedding_sequence(mrna_start_pos, len_seq, max_seq_len, d_model):
    """Generates positional embeddings for a sequence, padded to max_seq_len."""
    pe_list = []
    for offset in range(len_seq):
        t = mrna_start_pos + offset
        pe_list.extend([round(get_pos_embedding(i, d_model, t), 4) for i in range(d_model)])

    # Pad or truncate the final list to the expected length
    expected_len = max_seq_len * d_model
    if len(pe_list) < expected_len:
        pe_list.extend([0.0] * (expected_len - len(pe_list)))
    else:
        pe_list = pe_list[:expected_len]
    return pe_list


def build_kmers(sequence: str, k: int = 2) -> list:
    """Builds k-mers from a sequence."""
    if len(sequence) < k:
        return []
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def cal_thermo_feature(sequence: str, max_len_expected_output: int) -> list:
    """Calculates thermodynamic features for an siRNA sequence."""
    seq = sequence.upper().replace("T", "U")
    sum_stability = 0.0
    single_sum_values = []

    # Terminal nucleotide rules for siRNA
    if len(seq) > 0 and seq[0] == 'A': sum_stability += 0.45
    if len(seq) >= 19 and seq[18] == 'U': sum_stability += 0.45

    bimer_values_dict = {
        'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33, 'CU': -2.08,
        'AG': -2.08, 'CA': -2.11, 'UG': -2.11, 'GU': -2.24, 'AC': -2.24,
        'GA': -2.35, 'UC': -2.35, 'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42
    }
    bimers = build_kmers(seq, k=2)
    for b in bimers:
        stability_value = bimer_values_dict.get(b, 0.0)
        single_sum_values.append(stability_value)
        sum_stability += stability_value

    # Add corrections and final sum
    sum_stability += 4.09  # Intermolecular initiation
    sum_stability += 0.43  # Symmetry correction
    single_sum_values.append(round(sum_stability, 2))

    # Pad or truncate
    if len(single_sum_values) < max_len_expected_output:
        single_sum_values.extend([0.0] * (max_len_expected_output - len(single_sum_values)))
    else:
        single_sum_values = single_sum_values[:max_len_expected_output]
    return single_sum_values


def countGC(seq: str) -> float:
    """Calculates GC percentage. Returns 0.0 for empty sequences."""
    seq = seq.upper()
    if not seq:
        return 0.0
    return round((seq.count("G") + seq.count("C")) / len(seq), 3)


def get_kmer_freq(seq: str, k: int) -> list:
    """Calculates k-mer frequencies for all possible k-mers."""
    seq = seq.upper().replace("T", "U")
    bases = "ACGU"
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    kmer_freq = {kmer: 0.0 for kmer in all_kmers}

    total_kmers = len(seq) - k + 1
    if total_kmers <= 0:
        return list(kmer_freq.values())

    for i in range(total_kmers):
        kmer = seq[i:i + k]
        if kmer in kmer_freq:
            kmer_freq[kmer] += 1

    # Normalize frequencies
    for key in kmer_freq:
        kmer_freq[key] /= total_kmers
    return list(kmer_freq.values())

# --- K-mer wrappers for convenience ---
def single_freq(seq, *args): return get_kmer_freq(seq, 1)
def double_freq(seq, *args): return get_kmer_freq(seq, 2)
def triple_freq(seq, *args): return get_kmer_freq(seq, 3)
def quadruple_freq(seq, *args): return get_kmer_freq(seq, 4)
def quintuple_freq(seq, *args): return get_kmer_freq(seq, 5)


def one_hot_encode(score: int) -> list:
    """One-hot encodes a score of -1, 0, or 1."""
    if score == -1: return [1, 0, 0]
    if score == 0: return [0, 1, 0]
    if score == 1: return [0, 0, 1]
    return [0, 0, 0] # Fallback


def rules_scores(seq: str, max_rules_length: int = 19) -> list:
    """Obtains one-hot encoded positional scores for an siRNA sequence."""
    seq = seq.upper().replace("T", "U")
    one_hot_scores = []
    
    for i in range(max_rules_length):
        if i < len(seq) and i < len(position_scores):
            score = position_scores[i].get(seq[i], 0)
            one_hot_scores.extend(one_hot_encode(score))
        else:
            one_hot_scores.extend([0, 0, 0]) # Pad for shorter sequences
    
    # This final padding/truncation is redundant if max_rules_length is handled correctly
    # inside the loop, but it's kept as a safeguard.
    expected_len = max_rules_length * 3
    if len(one_hot_scores) < expected_len:
        one_hot_scores.extend([0] * (expected_len - len(one_hot_scores)))
    elif len(one_hot_scores) > expected_len:
        one_hot_scores = one_hot_scores[:expected_len]

    return one_hot_scores