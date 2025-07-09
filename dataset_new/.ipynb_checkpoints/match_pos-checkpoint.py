import pandas as pd
from Bio import SeqIO

# --- Load base data ---
#ff_df = pd.read_excel("viral_all_efficacy.xlsx")
eff_df = pd.read_csv("HIV_efficacy.csv")

# --- Load siRNA sequences ---
sirna_map = {
    rec.id: str(rec.seq).upper().replace("T", "U")
    for rec in SeqIO.parse("HIV_siRNA.fas", "fasta")
}
# --- Load mRNA sequences ---
mrna_map = {
    rec.id: str(rec.seq).upper().replace("T", "U")
    for rec in SeqIO.parse("mRNA_HIV_small.fas", "fasta")
}

# --- Add sequences ---
eff_df['siRNA_seq'] = eff_df['siRNA'].map(sirna_map)
eff_df['mRNA_seq_RNA-FM'] = eff_df['mRNA'].map(mrna_map)
eff_df['mRNA_seq'] = eff_df['mRNA_seq_RNA-FM'].str.replace("U", "T")


# Drop rows with missing mappings
eff_df = eff_df.dropna()

# --- Compute match position ---
c=0
def reverse_complement_rna(seq):
    return seq.strip().upper().translate(str.maketrans("AUGCN", "UACGN"))[::-1]

def find_match_pos(row):
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
eff_df[final_cols].to_csv("hiv_all_data.csv", index=False)

print(f"✅ Final merged dataset saved: {len(eff_df)} rows → viral_all_data.csv")  