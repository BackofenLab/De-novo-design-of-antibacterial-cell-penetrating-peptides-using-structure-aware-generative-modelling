"""
This script analysises the amino acid distribution of different datasets of peptides.
"""

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def aa_dist(dfs):
    """
    Computes the aa distribution in every dataframe given in the list.
    """
    distributions = []
    for df in dfs:
        count = Counter()

        sequences = df.values.flatten()
        for seq in sequences:
            if "B" in seq or "X" in seq or "Z" in seq:
                continue
            else:
                count.update(seq)
        distributions.append(count)
    return distributions


def plot_distributions(counters, labels=None):
    """
    Plots the aa distributions using matplotlb.
    """
    all_chars = set()
    for counter in counters:
        all_chars.update(counter.keys())
    
    all_chars = sorted(all_chars)
    max_percentage = max([max(counter.values()) / sum(counter.values()) for counter in counters])
    n = len(counters)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))

    if n == 1:
        axes = [axes]

    for i, (counter, ax) in enumerate(zip(counters, axes)):
        total_count = sum(counter.values())
        percentages = [(counter.get(char, 0) / total_count) * 100 for char in all_chars]
        ax.bar(all_chars, percentages, alpha=0.7)

        ax.set_xlabel('Character')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(labels[i] if labels else f'Counter {i+1}')
        ax.set_ylim(0, max_percentage * 100 + 5)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_grouped_distributions(counters, labels=None):
    """
    Plots amino acid distributions as a grouped bar chart for easier comparison.
    Uses fixed colors: blue, red, purple, green (with alpha=0.7).
    """
    # Collect all amino acids
    all_chars = sorted(set().union(*[c.keys() for c in counters]))
    
    # Compute percentages for each dataset
    percentages = []
    for counter in counters:
        total = sum(counter.values())
        percentages.append([(counter.get(char, 0) / total) * 100 for char in all_chars])
    
    # Number of datasets and positions
    n = len(counters)
    x = np.arange(len(all_chars))
    width = 0.7 / n  # narrower bars with spacing
    
    # Fixed colors in requested order
    fixed_colors = ["blue", "red", "purple", "green"]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, dist in enumerate(percentages):
        color = fixed_colors[i % len(fixed_colors)]
        ax.bar(x + i * width, dist, width,
               label=labels[i] if labels else f'Counter {i+1}',
               color=color, edgecolor="black", linewidth=0.4, alpha=0.7)
    
    # Style axes
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(all_chars, rotation=45, ha="right")
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Amino Acid Distributions", fontsize=14, pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Legend on top
    ax.legend(frameon=False, ncol=len(counters), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df1 = pd.read_csv("../Data/CPP_AAS.csv", header=None)
    df2 = pd.read_csv("../Data/Eval/VAE_FT_EVAL.csv", header=0)
    df3 = pd.read_csv("../Data/Eval/Similar_To_AAAAARRRIRKQAHAHSK_EVAL.csv", header=0)
    df4 = pd.read_csv("../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv", header=0)

    dfs = [df1, df2, df3, df4]
    titles = ["Validated_CPPs", "VAE_rand", "VAE_spiral", "VAE_string"]

    dists = aa_dist(dfs)
    # plot_distributions(dists, titles)
    plot_grouped_distributions(dists, titles)
