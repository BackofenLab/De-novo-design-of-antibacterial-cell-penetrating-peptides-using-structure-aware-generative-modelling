"""
Script attempting to cluster tertiary structures in automated fashion.
Not successfull and instead used manual classification (see classify_cpps.py)
"""

import ast
import numpy as np
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
import io
from io import StringIO
import base64
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser, Superimposer

from sklearn.cluster import AgglomerativeClustering


def extract_ca_coords(pdb_str):
    """
    Parses a PDB structure string, extracts the 3D coordinates of all alpha carbon 
    (CA) atoms, and returns them as a NumPy array.
    """
    parser = PDBParser(QUIET=True)
    handle = StringIO(pdb_str)
    structure = parser.get_structure("model", handle)
    coords = []

    for atom in structure.get_atoms():
        if atom.get_id() == "CA":
            coords.append(atom.get_coord())

    return np.array(coords)


def coords_to_image(coords, max_len=100):
    """
    Converts 3D coordinates into a compact line-plot image of the structure, 
    encodes it as a base64 PNG, and returns an HTML <img> tag for embedding.
    """
    coords = coords[:max_len]
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker='o', markersize=2)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"<img src='data:image/png;base64,{img_base64}'/>"


def compute_rmsd_matrix(df, column="pdb_structure"):
    """
    Computes a symmetric RMSD matrix for a set of protein structures by 
    extracting alpha carbon atoms, superimposing pairs of structures, and 
    recording their root-mean-square deviation.
    """
    parser = PDBParser(QUIET=True)
    n = len(df)
    rmsd_matrix = np.zeros((n, n))

    def extract_ca_atoms(pdb_str):
        structure = parser.get_structure("model", StringIO(pdb_str))
        return [atom for atom in structure.get_atoms() if atom.get_id() == "CA"]

    # Precompute all CA atoms once
    all_atoms = [extract_ca_atoms(pdb) for pdb in df[column]]

    for i in tqdm(range(n), desc="Generating RMSD matrix..."):
        atoms_i = all_atoms[i]
        for j in range(i + 1, n):
            atoms_j = all_atoms[j]
            min_len = min(len(atoms_i), len(atoms_j))
            if min_len == 0:
                rmsd = np.nan
            else:
                si = Superimposer()
                si.set_atoms(atoms_i[:min_len], atoms_j[:min_len])
                rmsd = si.rms
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

    return rmsd_matrix


def cluster(rmsd_matrix, threshold=3.0):
    """
    Performs agglomerative clustering on an RMSD matrix using average 
    linkage and a distance threshold, returning cluster assignments for all structures.
    """
    rmsd_clean = np.nan_to_num(rmsd_matrix, nan=np.nanmax(rmsd_matrix) + 10)
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold,
        n_clusters=None
    )
    return clustering.fit_predict(rmsd_clean)


if __name__ == "__main__":
    DATA_PATH = "../Data/CPP_AAS_with_structures.csv"
    
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH, header=0)
    
    matrix = compute_rmsd_matrix(df)
    labels = cluster(matrix)
    df["Cluster"] = labels

    df["Img"] = [coords_to_image(extract_ca_coords(c)) for c in tqdm(df["pdb_structure"], desc="Producing Images")]

    print("Saving to HTML...")
    df = df.sort_values("Cluster")
    df = df.drop(columns=["pdb_structure"])
    df.to_html("../Data/clustered_peptides.html", escape=False)
    
