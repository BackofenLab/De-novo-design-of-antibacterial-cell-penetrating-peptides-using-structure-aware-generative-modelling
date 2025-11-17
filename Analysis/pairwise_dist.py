"""
Script for computing the pairwise distances of different peptide 
datasets (and plotting them)
"""

import numpy as np
import pandas as pd
import parasail
import matplotlib.pyplot as plt
from tqdm import tqdm


def blosum62_distance(seq1, seq2):
    """
    Computes an alignment-based distance between two sequences using the 
    BLOSUM62 substitution matrix with affine gap penalties, returning the 
    alignment score.
    """
    # gap opening penalty: 10
    # gap extension penalty: 1
    # blosum62 substitution matrix
    result = parasail.nw_trace_striped_16(seq1, seq2, 10, 1, parasail.blosum62)
    return result.score


def compute_distance_matrix(seqs1, seqs2):
    """
    Computes a pairwise distance matrix between two sets of sequences 
    using BLOSUM62 alignment scores, iterating over all sequence pairs 
    and storing the distances in a NumPy array.
    """
    mat = np.zeros((len(seqs1), len(seqs2)))
    for i, s1 in enumerate(tqdm(seqs1, desc="Computing dist matrix")):
        for j, s2 in enumerate(seqs2):
            mat[i, j] = blosum62_distance(s1.replace(" ", ""), s2.replace(" ", ""))
    return mat


def plot_comparison_heatmaps(base_seqs, comp_seqs_list, labels):
    """
    Generates side-by-side heatmaps comparing validated CPP sequences 
    against multiple comparison sets, computes global color scaling for 
    consistency, and annotates each plot with summary statistics 
    (mean, median, standard deviation).
    """
    n = len(comp_seqs_list)
    distance_matrices = []
    vmin, vmax = float('inf'), float('-inf')

    # Compute all matrices first
    for seqs in comp_seqs_list:
        mat = compute_distance_matrix(base_seqs, seqs)
        distance_matrices.append(mat)

    # Compute global vmin/vmax, excluding self-comparison (assumed to be first)
    for mat, label in zip(distance_matrices[1:], labels[1:]):
        vmin = min(vmin, np.min(mat))
        vmax = max(vmax, np.max(mat))

    # Plot
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    for i, (mat, label) in enumerate(zip(distance_matrices, labels)):
        mean_val = np.mean(mat)
        median_val = np.median(mat)
        std_val = np.std(mat)
        title = f"Validated_CPP vs {label}\nMean: {mean_val:.2f}, Median: {median_val:.2f} Std: {std_val:.2f}"

        ax = axs[i]
        if i == 0:
            # Use individual scale for Validated vs Validated
            im = ax.imshow(mat, cmap="viridis", aspect='auto')
        else:
            # Use shared scale
            im = ax.imshow(mat, cmap="viridis", aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.set_ylabel("Validated_CPP")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df1 = pd.read_csv("../Data/CPP_AAS.csv", header=None)
    df2 = pd.read_csv("../Data/Eval/VAE_FT_EVAL.csv", header=0)
    df3 = pd.read_csv("../Data/Eval/Similar_To_AAAAARRRIRKQAHAHSK_EVAL.csv", header=0)
    df4 = pd.read_csv("../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv", header=0)
    df5 = pd.read_csv("../Data/Eval/UNIFORM_RANDOM_CPPs_EVAL.csv", header=0)

    base_seqs = df1.iloc[:, 0].tolist()
    dfs = [df1, df2, df3, df4, df5]
    labels = ["Validated_CPP", "CVAE_Rand", "CVAE_Spiral", "CVAE_String", "Uniform_Random"]
    comp_seqs_list = [df.iloc[:, 0].tolist() for df in dfs]

    plot_comparison_heatmaps(base_seqs, comp_seqs_list, labels)
