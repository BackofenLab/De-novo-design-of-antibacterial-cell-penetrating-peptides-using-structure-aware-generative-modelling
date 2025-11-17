"""
Script to analyse token frequency and entropy of a dataset.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import entropy
from collections import Counter


def load_data():
    """
    Loads sequences from a CSV file, splits entries in the "Codes" column 
    by #, and returns them as a list of tokenized sequences.
    """
    path = "../Data/TESTING.csv"
    df = pd.read_csv(path)
    return [seq.split("#") for seq in df["Codes"].dropna().tolist()]


def build_dict():
    """
    Builds a substitution dictionary from derivative data by mapping each 
    natural amino acid code to a list of related codes within the same family.
    """
    path = "../Data/SSD_Derivatives.txt"
    df = pd.read_csv(path, sep="\t")
    df = df[["Name", "L-Code", "SMILES", "family", "natural"]]


    nat_df = df[df["natural"] == "natural"]
    nat_map = dict(zip(nat_df["L-Code"], nat_df["SMILES"]))

    subs_map = {}
    for idx, (key, val) in enumerate(nat_map.items()):
        family = df[df["L-Code"] == key]["family"].tolist()[0]
        subs_map[key] = df[df["family"] == family]["L-Code"].tolist()

    return subs_map


def build_color_map(subs_map):
    """
    Creates a color mapping by assigning each substitution family a distinct 
    color from a categorical colormap, then mapping every code in the 
    families to its corresponding color.
    """
    all_codes = set(k for family in subs_map.values() for k in family)
    family_to_color = {tuple(sorted(v)): i for i, v in enumerate(subs_map.values())}
    color_palette = cm.get_cmap("tab20", len(family_to_color))
    
    code_to_color = {}
    for family, idx in family_to_color.items():
        for code in family:
            code_to_color[code] = color_palette(idx)
    return code_to_color


def token_frequency(data_flat):
    """
    Computes normalized token frequencies from a flattened sequence dataset 
    and returns them as a dictionary sorted in descending order.
    """
    counts = Counter(data_flat)
    total = sum(counts.values())
    freqs = {k: v / total for k, v in counts.items()}
    sorted_freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
    return sorted_freqs


def compute_entropy(data):
    """
    Calculates the positional entropy of sequences by counting token 
    distributions at each position, computing probability distributions, 
    and returning the entropy values per position.
    """
    max_len = max(len(x) for x in data)
    pos_counts = [Counter() for _ in range(max_len)]

    for seq in data:
        for i, tok in enumerate(seq[:max_len]):
            pos_counts[i][tok] += 1

    pos_entropy = []
    for ct in pos_counts:
        counts = np.array(list(ct.values()))
        prob = counts / counts.sum()
        pos_entropy.append(entropy(prob))

    return pos_entropy


if __name__ == "__main__":
    data = load_data()
    subs_map = build_dict()

    for key, value in subs_map.items():
        print("{}: {}".format(key, value))

    flat_data = [tok for seq in data for tok in seq]

    freqs = token_frequency(flat_data)
    entropy_vals = compute_entropy(data)

    code_to_color = build_color_map(subs_map)
    colors = [code_to_color.get(code, "gray") for code in freqs.keys()]

    fig, axs = plt.subplots(2, 1, figsize=(6, 5))

    # Token frequency with colored bars
    axs[0].bar(freqs.keys(), freqs.values(), color=colors)
    axs[0].set_title("Token Frequency Distribution")
    axs[0].set_ylabel("Frequency")
    axs[0].tick_params(axis='x', rotation=90)

    # Entropy per position
    axs[1].plot(entropy_vals, label="Observed Entropy")
    axs[1].axhline(y=math.log(20), color='red', linestyle='--', label="Max Entropy (ln(20))")
    axs[1].set_title("Token Entropy per Position")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Entropy")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

