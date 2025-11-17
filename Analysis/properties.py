"""
Script for computing the physicochemical properties of different 
datasets of peptides and comparing them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Descriptors import NumAromaticRings
from rdkit.Chem.Crippen import MolLogP


def amino_acid_properties(sequence: str):
    """
    Calculate molecular properties for a given amino acid sequence.

    :param sequence: Amino acid sequence.
    :return: Dictionary of molecular properties.
    """
    mol = Chem.MolFromSequence(sequence)
    if not mol:
        return {key: np.nan for key in ["MW", "tPSA", "Fsp3", "cLogP", "HBA", "HBD", "NAR", "NRB"]}

    property_calculators = {
        "MW": lambda m: Descriptors.MolWt(m),
        "tPSA": lambda m: Descriptors.TPSA(m),
        "Fsp3": lambda m: Descriptors.FractionCSP3(m),
        "cLogP": lambda m: Descriptors.MolLogP(m),
        "HBA": lambda m: Lipinski.NumHAcceptors(m),
        "HBD": lambda m: Lipinski.NumHDonors(m),
        "NAR": lambda m: Descriptors.NumAromaticRings(m),
        "NRB": lambda m: Lipinski.NumRotatableBonds(m),
    }

    properties = {}
    for prop, calculator in property_calculators.items():
        try:
            properties[prop] = calculator(mol)
        except:
            properties[prop] = np.nan

    return properties


def plot_properties(dataset_paths: list, headers: list = None, descriptions: list = None, colors: list = None):
    """
    Plots boxplots of physicochemical properties across multiple datasets, 
    computing metrics for each sequence, organizing results into subplots, 
    and coloring boxes per dataset with a legend for comparison.
    """
    metrics = ["MW", "tPSA", "Fsp3", "cLogP", "HBA", "HBD", "NAR", "NRB"]
    _metric = ["g/mol", "Å²", "Fraction", "logP", "Count", "Count", "Count", "Count"]

    # de Oliveira et al. (2021) value ranges for CPPs
    cpp_ranges = {
        "MW": (331.48, 3750.51),
        "cLogP": (-42.12, 2.97),
        "tPSA": (101.29, 1782.83),
        "Fsp3": (0.37, 0.84),
        "NRB": (9, 137),
        "HBD": (4, 69),
        "HBA": (5, 55),
        "NAR": (0, 10),
    }


    if headers == None:
        headers = [None for _ in range(len(dataset_paths))]

    if descriptions == None:
        descriptions = ["" for _ in range(len(dataset_paths))]

    if colors == None:
        colors = ["blue", "red", "purple", "green", "yellow"]

    all_datasets = []
    for idx, ds in enumerate(dataset_paths):
        df = pd.read_csv(ds, header=headers[idx])
        data = df.iloc[:, 0].tolist()

        dataset_properties = []
        for seq in data:
            props = amino_acid_properties(seq)
            props = [val for val in props.values()]
            row = [seq] + props
            dataset_properties.append(row)
        dataset_properties = pd.DataFrame(dataset_properties, columns=[["Sequence"] + metrics])
        all_datasets.append(dataset_properties)

    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
    fig.suptitle("Comparison of Metrics Across Datasets", fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i // 4, i % 4]
        data = [
            np.asarray(df[metric].dropna().values, dtype=np.float32).flatten()
            for df in all_datasets
        ]
        box = ax.boxplot(data, patch_artist=True)  # omit labels

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(metric)
        ax.set_xticks([])  # hide x-ticks
        ax.set_ylabel(_metric[i])

        if metric in cpp_ranges:
            lower, upper = cpp_ranges[metric]
            ax.axhline(lower, color="black", linestyle="dotted", linewidth=1)
            ax.axhline(upper, color="black", linestyle="dotted", linewidth=1)
            ax.fill_betweenx([lower, upper], x1=0, x2=len(all_datasets)+1, color="gray", alpha=0.1)

    # Build and add legend
    legend_patches = [
        mpatches.Patch(color=color, label=desc)
        for color, desc in zip(colors, descriptions)
    ]
    legend_patches.append(
        mpatches.Patch(
            facecolor="lightgray",
            edgecolor="black",
            linestyle="dotted",
            label="CPP range (de Oliveira et al., 2021)",
            alpha=0.4,
        )
    )
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 0.90))

    plt.tight_layout(rect=[0, 0.03, 0.82, 1])
    plt.show()


if __name__ == "__main__":
    paths = ["../Data/CPP_AAS.csv",
             "../Data/Eval/VAE_FT_EVAL.csv",
             "../Data/Eval/Similar_To_AAAAARRRIRKQAHAHSK_EVAL.csv",
             "../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv",
             "../Data/Eval/UNIFORM_RANDOM_CPPs_EVAL.csv"]
    headers = [None, 0, 0, 0, 0]
    descriptions = ["Validated CPPs", "TransformerVAE unconditioned", "TransformerVAE Spiral", "TransformerVAE String", "Random"]

    plot_properties(paths, headers, descriptions)

