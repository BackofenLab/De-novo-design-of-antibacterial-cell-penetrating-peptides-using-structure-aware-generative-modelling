"""
Script to analyse the lengths of peptides in varying datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def histograms(data_paths, tags):
    n = len(data_paths)
    fig, axs = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)

    for i, path in enumerate(data_paths):
        df = pd.read_csv(path)
        try:
            length_counts = df["Sequence"].str.len().value_counts().sort_index()
        except KeyError:
            length_counts = df.iloc[:, 0].str.len().value_counts().sort_index()

        axs[i].bar(length_counts.index, length_counts.values)
        axs[i].set_title(tags[i])
        axs[i].set_ylabel("Count")

    axs[-1].set_xlabel("CPP Length")
    plt.tight_layout()
    plt.show()


def violin_plots(data_paths, tags):
    lengths_list = []

    for path in data_paths:
        df = pd.read_csv(path)
        try:
            lengths = df["Sequence"].str.len()
        except KeyError:
            lengths = df.iloc[:, 0].str.len()
        lengths_list.append(lengths)

    fig, ax = plt.subplots(figsize=(15, 8))

    # violin plot (density shapes)
    vp = ax.violinplot(
        lengths_list,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # pastel colors
    colors = ["blue", "red", "purple", "green"]
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    # overlay narrow boxplot-style info
    box_width = 0.023
    whisker_width = 0.01

    for i, data in enumerate(lengths_list, start=1):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisk_low = np.min(data[data >= q1 - 1.5 * iqr])
        whisk_high = np.max(data[data <= q3 + 1.5 * iqr])

        # draw narrow box
        ax.add_patch(plt.Rectangle(
            (i - box_width, q1),   # (x, y)
            2 * box_width,         # width
            q3 - q1,               # height
            facecolor="white",
            edgecolor="black",
            lw=1.2,
            zorder=3
        ))

        # draw median line
        ax.hlines(med, i - box_width, i + box_width, color="black", lw=2, zorder=4)

        # draw whiskers
        ax.vlines(i, whisk_low, whisk_high, color="black", lw=1.2, zorder=3)
        ax.hlines([whisk_low, whisk_high], i - whisker_width, i + whisker_width, color="black", lw=1.2, zorder=3)

    ax.set_xticks(range(1, len(tags) + 1))
    ax.set_xticklabels(tags, rotation=20, fontsize=15)
    ax.set_ylabel("Length", fontsize=15)
    ax.set_title("Distributions of Peptide Lengths", fontsize=20, pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_paths = [
        "../Data/CPP_AAS.csv",
        "../Data/Eval/VAE_FT_EVAL.csv",
        "../Data/Eval/Similar_To_AAAAARRRIRKQAHAHSK_EVAL.csv",
        "../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv",
    ]

    tags = [
        "Validated CPPs",
        "VAE_rand",
        "VAE_spiral",
        "VAE_string",
    ]

    # histograms(data_paths, tags)
    violin_plots(data_paths, tags)

     

