"""
Script used to find the top k closest matches of a proposed peptide dataset
to validated CPPs. Initially we followed a KNN approach but then reverted to 
euclidean distance of physicochemical properties. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Descriptors import NumAromaticRings
from rdkit.Chem.Crippen import MolLogP

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


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
        try:            properties[prop] = calculator(mol)
        except:
            properties[prop] = np.nan

    return properties


def compute_properties(df: pd.DataFrame):
    """
    Computes physicochemical properties for each sequence in the dataframe, 
    aggregates them into a new dataframe with predefined metric columns, and 
    returns the result.
    """
    metrics = ["MW", "tPSA", "Fsp3", "cLogP", "HBA", "HBD", "NAR", "NRB"]

    data = df.iloc[:, 0].tolist()
    properties = []
    for seq in data:
        props = amino_acid_properties(seq)
        props = [val for val in props.values()]
        row = [seq] + props
        properties.append(row)
    properties = pd.DataFrame(properties, columns=[["Sequence"] + metrics])
    return properties                          



def find_best_matching_cpps(validated_df: pd.DataFrame, compare_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Identifies the top-k candidate CPPs from a comparison dataset that are 
    most similar to validated CPPs by computing physicochemical properties,
    normalizing features, measuring Euclidean distances, and returning a 
    combined dataframe of proposed and closest validated matches with distances.
    """
    # Assume sequences are in the first column
    validated_sequences = validated_df.iloc[:, 0]
    compare_filtered = compare_df[~compare_df.iloc[:, 0].isin(validated_sequences)].copy()

    # Compute properties and drop rows with NaNs
    validated_props = compute_properties(validated_df).dropna()
    compare_props = compute_properties(compare_filtered).dropna()
    metrics = ["MW", "tPSA", "Fsp3", "cLogP", "HBA", "HBD", "NAR", "NRB"]

    # Normalize features
    scaler = StandardScaler()
    scaler.fit(pd.concat([validated_props[metrics], compare_props[metrics]]))
    X_validated = scaler.transform(validated_props[metrics])
    X_compare = scaler.transform(compare_props[metrics])

    # Compute distances
    dist_matrix = pairwise_distances(X_compare, X_validated, metric="euclidean")
    min_distances = dist_matrix.min(axis=1)
    closest_indices = dist_matrix.argmin(axis=1)

    # Select top-k closest compare CPPs
    top_indices = np.argsort(min_distances)[:top_k]
    top_matches = compare_props.iloc[top_indices].copy()
    top_matches["MinDistance"] = min_distances[top_indices]

    # Get closest validated entries for those top-k compare CPPs
    validated_matches = validated_props.iloc[closest_indices[top_indices]].reset_index(drop=True)
    validated_sequences = validated_df.iloc[validated_props.index].reset_index(drop=True)
    validated_matches["Sequence"] = validated_sequences.iloc[closest_indices[top_indices], 0].values  # assumes first col is sequence

    # Rename columns to distinguish
    top_matches = top_matches.reset_index(drop=True)
    top_matches = top_matches.add_prefix("Proposed_CPP_")
    validated_matches = validated_matches.add_prefix("Validated_CPP_")

    # Combine into one DataFrame
    combined = pd.concat([top_matches, validated_matches], axis=1)

    combined.rename(columns={"Proposed_CPP_MinDistance": "Distance"}, inplace=True)

    return combined


def plot_panel_a(validated_df, compare_df, top_k=10):
    combined = find_best_matching_cpps(validated_df, compare_df, top_k=top_k)
    print(combined.columns)

    seqs = combined["Proposed_CPP_Sequence"]
    val_seqs = combined["Validated_CPP_Sequence"]
    dist = combined["Distance"]


def plot_panel_radar(validated_df, compare_df, top_k=10):
    metrics = ["MW", "tPSA", "Fsp3", "cLogP", "HBA", "HBD", "NAR", "NRB"]
    combined = find_best_matching_cpps(validated_df, compare_df, top_k=top_k)
    combined.columns = combined.columns.get_level_values(0)

    P = combined[[f"Proposed_CPP_{m}" for m in metrics]].rename(columns=lambda c: c.replace("Proposed_CPP_", ""))
    V = combined[[f"Validated_CPP_{m}" for m in metrics]].rename(columns=lambda c: c.replace("Validated_CPP_", ""))
    proposed_names = combined["Proposed_CPP_Sequence"].astype(str).tolist()
    validated_names = combined["Validated_CPP_Sequence"].astype(str).tolist()

    X = pd.concat([P, V], axis=0)
    mins, maxs = X.min(), X.max()
    rngs = maxs - mins

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    n = min(len(proposed_names), 10)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True), figsize=(cols * 4.5 + 1.8, rows * 4.5))
    axes = np.array(axes).reshape(-1)

    color_proposed = "#1f77b4"
    color_validated = "#ff7f0e"

    for i in range(n):
        ax = axes[i]
        pname, vname = proposed_names[i], validated_names[i]
        pv, vv = P.iloc[i].tolist(), V.iloc[i].tolist()

        pv_scaled = [(pv[j] - mins[j]) / rngs[j] if rngs[j] != 0 else 0.5 for j in range(len(metrics))]
        vv_scaled = [(vv[j] - mins[j]) / rngs[j] if rngs[j] != 0 else 0.5 for j in range(len(metrics))]
        pv_scaled += pv_scaled[:1]
        vv_scaled += vv_scaled[:1]

        ax.plot(angles, pv_scaled, linewidth=2, color=color_proposed)
        ax.fill(angles, pv_scaled, alpha=0.18, color=color_proposed)
        ax.plot(angles, vv_scaled, linewidth=2, color=color_validated)
        ax.fill(angles, vv_scaled, alpha=0.18, color=color_validated)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.35)
        ax.tick_params(axis='both', labelsize=7)

        for j, a in enumerate(angles[:-1]):
            for frac in (0.25, 0.5, 0.75, 1.0):
                val = mins[j] + frac * rngs[j]
                ax.text(a, frac, f"{val:.1f}", fontsize=6, ha="center", va="center", color="#666666")

        ax.text(0.5, 1.22, f"Proposed: {pname}", color=color_proposed,
                ha="center", va="center", fontsize=9, transform=ax.transAxes, fontweight="bold")
        ax.text(0.5, 1.13, f"Validated: {vname}", color=color_validated,
                ha="center", va="center", fontsize=9, transform=ax.transAxes)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [
        Line2D([0], [0], color=color_proposed, lw=2, label="Proposed"),
        Line2D([0], [0], color=color_validated, lw=2, label="Validated"),
    ]
    plt.suptitle("Top 10 Proposed CPPs (String) Closest to Validated Counterparts in Physicochemical Space", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.94, 0.97])
    plt.show()


if __name__ == "__main__":
    validated_cpps = pd.read_csv("../Data/CPP_AAS.csv", header=None)
    compare_cpps = pd.read_csv("../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv", header=0)

    """
    res = find_best_matching_cpps(validated_cpps, compare_cpps, top_k=30) 
    
    for idx, row in res.iterrows():
        print("{} - {} - {}".format(row["Proposed_CPP_Sequence"], row["Distance"], row["Validated_CPP_Sequence"]))

    res.to_csv("../Data/Top_k/String_top_30.csv", index=False)
    """
    plot_panel_radar(validated_cpps, compare_cpps)
