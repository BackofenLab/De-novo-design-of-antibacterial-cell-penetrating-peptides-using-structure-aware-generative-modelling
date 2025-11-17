import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def plot_hist(new_df, savepath="./DATASET_HISTOGRAM_FIGURE.png"):
    sns.set_theme(style="whitegrid", font="Helvetica", font_scale=1.0)
    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300
    })

    # --- Create subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # (a) Sequence lengths
    sns.histplot(new_df["Lengths"], bins=25, color="#8EC6E6", ax=axes[0])
    axes[0].set_title("a) Sequence length", loc="left", fontsize=10)
    axes[0].set_xlabel("Length (amino acids)")
    axes[0].set_ylabel("Count")

    # (b) Charge distribution
    sns.histplot(new_df["Charge"], bins=25, color="#A0D8A0", ax=axes[1])
    axes[1].set_title("b) Net charge", loc="left", fontsize=10)
    axes[1].set_xlabel("Net charge at pH 7.0")
    axes[1].set_ylabel("")

    # (c) Structural class distribution
    sns.countplot(y="Structure", data=new_df, palette=["#F7B2BD"], ax=axes[2])
    axes[2].set_title("c) Structural class", loc="left", fontsize=10)
    axes[2].set_xlabel("Count")
    axes[2].set_ylabel("")

    # --- Layout & export ---
    plt.tight_layout(pad=1)
    plt.savefig(savepath, dpi=600, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"Saved histograms to {savepath}") 


if __name__ == "__main__":
    df = pd.read_csv("../Data/classified_structures.csv")

    charges = []
    for idx, row in df.iterrows():
        aa = row["AA"]
        chge = ProteinAnalysis(aa)
        aa_len = len(aa)
        structure = row["label"]

        charges.append([aa, chge.charge_at_pH(7.0), aa_len, structure])
    new_df = pd.DataFrame(charges, columns=["CPP", "Charge", "Lengths", "Structure"])
    new_df["Structure"] = (
        new_df["Structure"]
        .fillna("Other")
        .astype(str)
        .replace("", "Other")
    )
    print(new_df["Structure"].value_counts())
    print(new_df["Structure"])
    plot_hist(new_df)
