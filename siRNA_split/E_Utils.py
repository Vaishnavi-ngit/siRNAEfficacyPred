import numpy
import pandas
import math
import torch
from transformers import AutoTokenizer, AutoModel, logging

# Suppress warnings from transformers library
logging.set_verbosity_error()

# --- Global Model and Tokenizer Initialization (LOADED ONCE) ---
# Define the model name
MP_RNA_MODEL_NAME = "yangheng/MP-RNA"

# Initialize global variables for tokenizer and model
GLOBAL_MP_RNA_TOKENIZER = None
GLOBAL_MP_RNA_MODEL = None

# Determine the device (CPU or GPU)
GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    print(f"E_Utils: Attempting to load MP-RNA Transformer tokenizer and model on {GLOBAL_DEVICE}...")
    GLOBAL_MP_RNA_TOKENIZER = AutoTokenizer.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True)
    GLOBAL_MP_RNA_MODEL = AutoModel.from_pretrained(MP_RNA_MODEL_NAME, trust_remote_code=True).to(GLOBAL_DEVICE)
    GLOBAL_MP_RNA_MODEL.eval() # Set model to evaluation mode
    print("E_Utils: MP-RNA Transformer loaded successfully.")
except Exception as e:
    print(f"E_Utils: ERROR: Could not load MP-RNA Transformer model '{MP_RNA_MODEL_NAME}'.")
    print(f"E_Utils: Please ensure you have internet access for the first download or the model is cached.")
    print(f"E_Utils: Error details: {e}")
    print("E_Utils: MP-RNA Transformer will not be available for embedding generation. Returning zero embeddings.")
    # GLOBAL_MP_RNA_TOKENIZER = None
    # GLOBAL_MP_RNA_MODEL = None
# --- End Global Model Initialization ---


# Position-specific scores for 19-nucleotide siRNA rules
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

MP_RNA_EMBEDDING_DIM = 768

def get_mp_rna_sequence_embedding(seq: str) -> numpy.ndarray:
    """
    Generates a sequence embedding using the pre-trained MP-RNA model.
    Model and tokenizer are loaded globally to ensure efficiency.
    """
    # If model/tokenizer failed to load globally, or if sequence is empty/invalid, return zeros
    if GLOBAL_MP_RNA_MODEL is None or GLOBAL_MP_RNA_TOKENIZER is None or not isinstance(seq, str) or not seq:
        return numpy.zeros(MP_RNA_EMBEDDING_DIM, dtype=numpy.float32)

    # Tokenize the sequence
    # Added truncation and padding for Transformer models. Adjust max_length as needed.
    inputs = GLOBAL_MP_RNA_TOKENIZER(seq, return_tensors="pt", truncation=True, padding='max_length', max_length=128)

    # Ensure inputs are on the same device as the model (CPU or CUDA)
    inputs = {k: v.to(GLOBAL_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = GLOBAL_MP_RNA_MODEL(**inputs)

    # Extract [CLS] token embedding, move to CPU, and convert to numpy
    sequence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    return sequence_embedding

def get_pos_embedding(index, d_model, t):
    """Calculates a single value for positional encoding."""
    if index % 2 == 0:
        return math.sin(t / pow(10000, index / (d_model - 1)))
    else:
        return math.cos(t / pow(10000, index / (d_model - 1)))

def get_pos_embedding_sequence(mrna_start_pos, len_sirna, d_model):
    """Generates a sequence of positional embeddings for the interaction."""
    pe_list = []
    # If mrna_start_pos is -1 (no match), return zeros for all features
    if mrna_start_pos == -1:
        return [0.0] * (len_sirna * d_model) # Return correct dimensionality of zeros

    for offset in range(len_sirna):
        t = mrna_start_pos + offset
        for i in range(d_model):
            pe = get_pos_embedding(i, d_model, t)
            pe_list.append(pe)
    return pe_list

def build_kmers(sequence):
    """Builds a list of 2-mer (bimer) sequences from a given sequence."""
    kmers = []
    if len(sequence) < 2: # Ensure sequence is long enough for at least one bimer
        return []
    n_kmers = len(sequence) - 1

    for i in range(n_kmers):
        kmer = sequence[i:i + 2]
        kmers.append(kmer)
    return kmers

def cal_thermo_feature(sequence, intermolecular_initiation=4.09, symmetry_correction=0.43):
    """
    Calculates individual bimer stabilities and total sum stability for an RNA sequence.
    Corrected for 19-mer siRNA to correctly apply the 3' U initiation rule at index 18.
    """
    sequence = sequence.upper().replace("T", "U")

    sum_stability = 0
    single_sum = [] # This list will hold individual bimer stabilities and the total sum

    # Handle very short sequences gracefully for thermodynamics.
    # If a sequence is too short to form any bimers, return a list of zeros.
    # For a 19-mer, there are 18 bimers + 1 total sum = 19 features.
    if len(sequence) < 2:
        return [0.0] * 19 # Return zeros for all expected bimer and total features

    bimers = build_kmers(sequence)

    # Initiation energies for terminal bases
    # For a 19-mer, 5' A is at index 0, and 3' U is at index 18.
    if len(sequence) >= 1 and sequence[0] == 'A':
        sum_stability += 0.45
    if len(sequence) >= 19 and sequence[18] == 'U': # Check for 19-mer length and 3' U
        sum_stability += 0.45

    bimer_values = {
        'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33,
        'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11,
        'GU': -2.24, 'AC': -2.24, 'GA': -2.35, 'UC': -2.35,
        'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42
    }
    for b in bimers:
        stability_value = bimer_values.get(b, 0)
        single_sum.append(round(stability_value, 3)) # Round individual bimer stabilities
        sum_stability += stability_value

    sum_stability += intermolecular_initiation
    sum_stability += symmetry_correction
    single_sum.append(round(sum_stability, 3)) # Round total sum stability

    # Pad with zeros if the sequence was shorter than 19 bases but had some bimers,
    # to maintain a consistent feature vector length.
    while len(single_sum) < 19:
        single_sum.insert(0, 0.0) # Prepend zeros
    return single_sum

def countGC(seq):
    """Calculates the GC content percentage of a sequence."""
    seq_processed = seq.upper().replace("T","U")
    if not seq_processed: # Handle empty sequence to prevent ZeroDivisionError
        return 0.0
    gc_percent = (seq_processed.count("G") + seq_processed.count("C")) / len(seq_processed)
    return round(gc_percent, 3)

def countAU(seq):
    """Calculates the AU content percentage of a sequence. (NEW)"""
    seq_processed = seq.upper().replace("T", "U")
    if not seq_processed: # Handle empty sequence
        return 0.0
    au_percent = (seq_processed.count("A") + seq_processed.count("U")) / len(seq_processed)
    return round(au_percent, 3)

def get_regional_gc_content(seq, window_size=5, overlap=0):
    """
    Computes the GC content for overlapping or non-overlapping sliding windows. (NEW)
    Returns a list of GC percentages for each window.
    """
    seq_processed = seq.upper().replace("T", "U")
    regional_gc = []
    step = window_size - overlap
    
    if step <= 0:
        # Default to a safe step (e.g., 1) if calculated step is invalid
        step = 1
        
    # Calculate expected number of windows for a 19-mer with specified window_size/step
    # This is to ensure a consistent output feature dimension
    num_expected_windows_for_19mer = math.ceil(max(0, 19 - window_size + 1) / step) if step > 0 else 0
    if num_expected_windows_for_19mer == 0: num_expected_windows_for_19mer = 1 # At least one feature

    if len(seq_processed) < window_size: # Handle sequences shorter than window_size
        return [0.0] * num_expected_windows_for_19mer # Return zeros for the expected number of features

    for i in range(0, len(seq_processed) - window_size + 1, step):
        window = seq_processed[i : i + window_size]
        if not window: # Should not happen if len(seq_processed) >= window_size
            regional_gc.append(0.0)
            continue
        gc_percent = (window.count("G") + window.count("C")) / len(window)
        regional_gc.append(round(gc_percent, 3))
        
    # Pad with zeros if the sequence was valid but somehow produced fewer windows than expected for a 19-mer
    while len(regional_gc) < num_expected_windows_for_19mer:
        regional_gc.append(0.0)

    return regional_gc

def get_terminal_base_one_hot(seq, length=19):
    """
    Generates a one-hot encoding for the nucleotide bases at the 5' and 3' ends
    of the sequence, specifically for a sequence of 'length' (19 for siRNA). (NEW)
    Returns a list of 8 floats (4 for 5' base, 4 for 3' base).
    """
    seq_processed = seq.upper().replace("T", "U")
    # Mapping for one-hot encoding A, C, G, U
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
    default_one_hot = [0, 0, 0, 0] # For missing or invalid bases

    terminal_features = []
    
    # 5' end base (first base, index 0)
    if len(seq_processed) > 0:
        terminal_features.extend(mapping.get(seq_processed[0], default_one_hot))
    else:
        terminal_features.extend(default_one_hot) # Default for empty sequence

    # 3' end base (last base at the specified 'length', index 'length - 1')
    if length > 0 and len(seq_processed) >= length:
        terminal_features.extend(mapping.get(seq_processed[length - 1], default_one_hot))
    else:
        terminal_features.extend(default_one_hot) # Default if sequence is too short or length is invalid

    return terminal_features

def get_thermo_asymmetry(sequence, region1_len=5, region2_len=5):
    """
    Calculates a simplified thermodynamic asymmetry score by comparing the summed stability
    of a 5' end region to a 3' end region of the siRNA duplex. (NEW)
    """
    # cal_thermo_feature already handles T to U conversion
    # It returns [bimer_1_stability, ..., bimer_18_stability, total_sum_stability]
    all_thermo_features = cal_thermo_feature(sequence)
    
    # If cal_thermo_feature returned all zeros (due to very short sequence), handle it
    if all(x == 0.0 for x in all_thermo_features):
        return 0.0
        
    # We are interested in individual bimer stabilities, which are all but the last element
    bimer_stabilities = all_thermo_features[:-1] # This will be 18 bimers for a 19-mer

    sum_stability_5_prime = 0.0
    # Sum bimers from the 5' end. `region1_len` bases will correspond to `region1_len - 1` bimers.
    if len(bimer_stabilities) >= (region1_len - 1): 
        sum_stability_5_prime = sum(bimer_stabilities[i] for i in range(min(region1_len - 1, len(bimer_stabilities))))

    sum_stability_3_prime = 0.0
    # Sum bimers from the 3' end. `region2_len` bases will correspond to `region2_len - 1` bimers.
    # Start index for 3' region bimers: (total_bimers - (region2_len - 1))
    start_index_3_prime = len(bimer_stabilities) - (region2_len - 1)
    if start_index_3_prime >= 0:
        sum_stability_3_prime = sum(bimer_stabilities[i] for i in range(max(0, start_index_3_prime), len(bimer_stabilities)))
    
    asymmetry_score = sum_stability_5_prime - sum_stability_3_prime
    return round(asymmetry_score, 3)

def single_freq(seq):
    """Calculates single nucleotide frequencies (A, C, G, U)."""
    seq_processed = seq.upper().replace("T","U")
    if not seq_processed:
        return {base: 0.0 for base in "ACGU"}
    freq = {base: round(seq_processed.count(base) / len(seq_processed), 3) for base in "ACGU"}
    return freq

def double_freq(seq):
    """Calculates 2-mer (bimer) frequencies."""
    seq_processed = seq.upper().replace("T","U")
    all_double_base = [a + b for a in "ACGU" for b in "ACGU"]
    double_base_freq = {dbase: 0 for dbase in all_double_base}
    total_mers = len(seq_processed) - 1

    if total_mers <= 0: # Handle sequences too short for 2-mers
        return {key : 0.0 for key in all_double_base}

    for i in range(total_mers):
        double_base = seq_processed[i:i+2]
        if double_base in double_base_freq: # Only count valid 2-mers
            double_base_freq[double_base] += 1

    double_base_freq = {key : round(value / total_mers, 3) for key, value in double_base_freq.items()}
    return double_base_freq

def triple_freq(seq):
    """Calculates 3-mer frequencies."""
    seq_processed = seq.upper().replace("T","U")
    all_triple_base = [a + b + c for a in "ACGU" for b in "ACGU" for c in "ACGU"] # Corrected to 3 loops
    triple_base_freq = {tbase: 0 for tbase in all_triple_base}
    total_mers = len(seq_processed) - 2

    if total_mers <= 0: # Handle sequences too short for 3-mers
        return {key : 0.0 for key in all_triple_base}

    for i in range(total_mers):
        triple_base = seq_processed[i:i+3]
        if triple_base in triple_base_freq:
            triple_base_freq[triple_base] += 1

    triple_base_freq = {key : round(value / total_mers, 3) for key, value in triple_base_freq.items()}
    return triple_base_freq

def quadruple_freq(seq):
    """Calculates 4-mer frequencies."""
    seq_processed = seq.upper().replace("T","U")
    all_quadruple_base = [a + b + c + d for a in "ACGU" for b in "ACGU" for c in "ACGU" for d in "ACGU"]
    quadruple_base_freq = {tbase: 0 for tbase in all_quadruple_base}
    total_mers = len(seq_processed) - 3

    if total_mers <= 0: # Handle sequences too short for 4-mers
        return {key : 0.0 for key in all_quadruple_base}

    for i in range(total_mers):
        quadruple_base = seq_processed[i:i+4]
        if quadruple_base in quadruple_base_freq:
            quadruple_base_freq[quadruple_base] += 1

    quadruple_base_freq = {key : round(value / total_mers, 3) for key, value in quadruple_base_freq.items()}
    return quadruple_base_freq

def quintuple_freq(seq):
    """Calculates 5-mer frequencies."""
    seq_processed = seq.upper().replace("T","U")
    all_quintuple_base = [a + b + c + d + e for a in "ACGU" for b in "ACGU" for c in "ACGU" for d in "ACGU" for e in "ACGU"]
    quintuple_base_freq = {tbase: 0 for tbase in all_quintuple_base}
    total_mers = len(seq_processed) - 4

    if total_mers <= 0: # Handle sequences too short for 5-mers
        return {key : 0.0 for key in all_quintuple_base}

    for i in range(total_mers):
        quintuple_base = seq_processed[i:i+5]
        if quintuple_base in quintuple_base_freq:
            quintuple_base_freq[quintuple_base] += 1

    quintuple_base_freq = {key : round(value / total_mers, 3) for key, value in quintuple_base_freq.items()}
    return quintuple_base_freq

def one_hot_encode(score):
    """Helper to one-hot encode rule scores (-1, 0, 1)."""
    if score == -1:
        return [1, 0, 0]
    elif score == 0:
        return [0, 1, 0]
    elif score == 1:
        return [0, 0, 1]
    else:
        # Default to neutral if an unexpected score comes up
        return [0, 1, 0]

def rules_scores(seq):
    """
    Applies siRNA design rules to a sequence and returns a one-hot encoded
    vector of scores. Fixed for 19-mer siRNA length.
    """
    seq_processed = seq.upper().replace("T","U")
    one_hot_scores = []
    # Iterate exactly 19 times for 19-mer rules
    for i in range(19):
        # Ensure sequence is long enough for the position
        if i < len(seq_processed):
            score = position_scores[i].get(seq_processed[i], 0) # Use .get for robustness against invalid bases
            one_hot_scores.extend(one_hot_encode(score))
        else:
            # If sequence is shorter than 19, pad with neutral scores
            one_hot_scores.extend([0, 1, 0])
    return one_hot_scores