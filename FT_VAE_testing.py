"""
Script for conducting a bunch of tests on the trained models (TransformerVAE, 
PeptideEmbedNet and CoordVAE)
"""

import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from transformers import AutoTokenizer, EsmConfig
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.functional import kl_div, log_softmax, softmax

from Model.model import PeptideEmbedNet
from Model.vae import TransformerVAE
from Model.TSC_vae import CoordVAE
from globals import device, tokenizer


def setup_model(latent_dim):
    """
    Initializes TransformerVAE by loading pretrained embedding and 
    structure models, restoring the main model weights from checkpoints, 
    setting all components to evaluation mode, and returning the assembled model.
    """
    print("Setting up Model...")
    main_weight_path = "./Checkpoints/2025_07_05_VAE/vae.pth"
    weight_path = "./Checkpoints/2025_07_03_MLM/model_20250703_134624.pth"
    ts_weight_path = "./Checkpoints/2025_07_05_TS/ts_vae.pt"

    pt_model = PeptideEmbedNet(num_layers=1,
                               ffn_dim=512,
                               context_window=64,
                               causal_masking=False)
    pt_model.load_state_dict(torch.load(weight_path))
    pt_model.eval()

    ts_model = CoordVAE(latent_dim=latent_dim)
    ts_model.load_state_dict(torch.load(ts_weight_path))
    ts_model.eval()

    model = TransformerVAE(
        pretrained_embedding_model=pt_model,
        pretrained_structure_model=ts_model,
        latent_dim=latent_dim,
        vocab_size=tokenizer.vocab_size,
        num_heads=4,
        max_len=100
    )
    model.load_state_dict(torch.load(main_weight_path))

    return model


def codes_to_aa(seq: str):
    """
    Converts a space-separated sequence of three-letter amino acid codes 
    into a one-letter sequence, returning None if an unknown code is encountered 
    or if the resulting sequence is shorter than five residues.
    """
    sequence = seq.split(" ")

    map = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }
    map = {v: k for k, v in map.items()}
    
    try:
        new_seq = [map[k] for k in sequence]

        if len(new_seq) < 5:
            return None

    except KeyError:
        return None

    return "".join(new_seq)


def generate_similar_aas(model, df, row_idx=0, amount=10, to_console=False, temperature=1, random_sample=False):
    """
    Generates a specified number of amino acid sequences similar to a target structure 
    by sampling from the model’s latent space, filters out invalid outputs, 
    optionally logs results to the console, and returns the sequences as a 
    dataframe along with a descriptive label.
    """
    test_aa, test_pdb, _ = get_row_data(df, row_idx)

    similar_aas = []
    while len(similar_aas) < amount:
        if random_sample:
            z1 = torch.zeros(1, model.latent_dim).to(device)
            z2 = torch.randn(1, model.latent_dim).to(device)
            new_aa = check_structure(model, test_pdb, temperature=temperature, z_s=[z1, z2])
        else:
            new_aa = check_structure(model, test_pdb, temperature=temperature)

        if new_aa != None:
            similar_aas.append(new_aa)

    if to_console:
        print("Input tertiary structure from: ")
        print(test_aa)
        print("")

        print("Proposed Similar AAs: ")
        print(similar_aas)

    similar_aas = pd.DataFrame(similar_aas, columns=[["Sequence"]])

    if random_sample:
        desc = "VAE_FT_EVAL"
    else:
        desc = "Similar_To_{}_EVAL".format(test_aa)

    return similar_aas, desc


def check_structure(model, pdb, temperature=1, z_s=None):
    """
    Decodes a protein sequence from latent vectors (sampled from 
    the model or provided), applies temperature-scaled sampling until 
    the [SEP] token, converts the output into one-letter amino acid codes, 
    and returns the resulting sequence.
    """
    if z_s is None:
        z1, z2 = get_samples(model, pdb) 
    else:
        z1 = z_s[0]
        z2 = z_s[1]

    logits = model.decode(z1, z2, tgt_len=100)
    
    probs = F.softmax(logits / temperature, dim=-1)
    sampled_ids = torch.multinomial(probs.squeeze(0), num_samples=1).squeeze(1).tolist()

    sep_token_id = tokenizer.sep_token_id
    if sep_token_id in sampled_ids:
        sep_index = sampled_ids.index(sep_token_id)
        sampled_ids = sampled_ids[:sep_index]

    decoded_seq = tokenizer.decode(sampled_ids, skip_special_tokens=True)
    return codes_to_aa(decoded_seq)


def get_samples(model, pdb):
    """
    Generates latent samples by setting the sequence latent vector to 
    zeros and deriving the structure latent vector from a given PDB, 
    then returns both for decoding.
    """
    z1 = torch.zeros(1, model.latent_dim).to(device)  # zeros can be replaced by randn
    z2 = get_struct_embedding(model, pdb)

    return z1, z2


def get_struct_embedding(model, pdb):
    """
    Encodes a PDB structure with CoordVAE, samples multiple 
    latent vectors via reparameterization, selects the one 
    with the highest log-probability under the posterior, and 
    returns it as the structure embedding.
    """
    inp_struct = np.array([CoordVAE.preprocess_coords(pdb)])
    inp_struct = torch.tensor(inp_struct, dtype=torch.float32)
    
    with torch.no_grad():
        mu, logvar = model.ts_model.encode(inp_struct)
    
    best_z = model.ts_model.reparameterize(mu, logvar)
    best_log_prob = log_gaussian(best_z, mu, logvar)
    
    # get good sample for distribution
    for _ in range(10):
        z = model.ts_model.reparameterize(mu, logvar)
        log_prob = log_gaussian(z, mu, logvar)

        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_z = z
    
    return best_z


def log_gaussian(x, mu, logvar):
    """
    Computes the mean log-probability of a sample under a Gaussian 
    distribution defined by given mean and log-variance, measuring 
    how well the sample fits the distribution.
    """
    sigma = torch.exp(0.5 * logvar)
    normal_dist = torch.distributions.Normal(mu, sigma)
    log_prob = normal_dist.log_prob(x)
    return log_prob.mean()


def get_row_data(df, row_id):
    """
    Retrieves amino acid sequence, PDB structure, and code values 
    from a specific row of the dataframe and returns them as a tuple.
    """
    row = df.iloc[row_id]
    aa = row["AA"]
    pdb = row["pdb_structure"]
    code = row["Codes"]

    return aa, pdb, code


def compare_structure_pairs(model, df, temperature=1):
    """
    Compares all sequence–structure pairs in a dataset by decoding 
    with the model, caching probabilities and latent embeddings, then 
    computing KL divergence between output distributions and L2 distances 
    between structure embeddings for every pair, returning the results as
    a list of comparisons.
    """
    results = []
    data = []
    for i in tqdm(range(len(df)), desc="Precomputing"):
        aa, pdb, _ = get_row_data(df, i)
        z1, z2 = get_samples(model, pdb) 

        out = model.decode(z1, z2, tgt_len=100)
        log_probs = log_softmax(out / temperature, dim=-1)
        probs = softmax(out / temperature, dim=-1)

        data.append({
            "aa": aa,
            "z2": z2,
            "log_probs": log_probs,
            "probs": probs
        })

    # Step 2: Pairwise comparison using cached data
    for i in tqdm(range(len(data)), desc="Comparing"):
        for j in range(len(data)):
            kl_ij = kl_div(data[i]["log_probs"], data[j]["probs"], reduction="batchmean").item()
            l2 = torch.norm(data[i]["z2"] - data[j]["z2"], p=2).item()

            results.append({
                "KL(A||B)": round(kl_ij, 4),
                "Latent L2": round(l2, 4),
                "pair": f"{data[i]['aa']} : {data[j]['aa']}"
            })

    return results


def plot_with_fit(results):
    """
    Plots KL divergence against latent L2 distances from pairwise comparisons, 
    adds a linear regression fit, computes and displays the Pearson correlation, 
    and visualizes the relationship with a scatter plot.
    """
    l2s = np.array([r["Latent L2"] for r in results])
    kls = np.array([r["KL(A||B)"] for r in results])

    plt.figure(figsize=(10, 5))
    plt.scatter(l2s, kls, alpha=0.6)

    # Fit and plot regression line
    m, b = np.polyfit(l2s, kls, deg=1)
    plt.plot(l2s, m * l2s + b, color="red", linewidth=2, label="Linear fit")

    # Correlation
    corr, _ = pearsonr(l2s, kls)
    plt.title(f"KL vs Latent Structure Distance (r = {corr:.2f})")
    plt.xlabel("Latent L2 Distance (||z2_i - z2_j||)")
    plt.ylabel("KL Divergence (A || B)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_distributions(model, df, row_id_0, row_id_1):
    """
    Compares the structural latent distributions of two sequences by
    retrieving their mean and log-variance from the model and visualizing 
    them with a distribution plot.
    """
    aa0, pdb0, _ = get_row_data(df, row_id_0)
    aa1, pdb1, _ = get_row_data(df, row_id_1)

    mu0, logvar0 = get_struct_distribution(model, pdb0)
    mu1, logvar1 = get_struct_distribution(model, pdb1)

    plot_normal_distributions(mu0, logvar0, mu1, logvar1)


def rank_dist(model, df, row_id):
    """
    Ranks the latent distribution dimensions of a structure by the 
    absolute magnitude of their mean values, returning an ordered 
    mapping of indices to mean values.
    """
    aa, pdb, _ = get_row_data(df, row_id)
    mu, logvar = get_struct_distribution(model, pdb)

    abs_mu = {idx: (entry, abs(entry)) for idx, entry in enumerate(mu[0].tolist())}
    abs_mu = {idx: value[0] for idx, value in sorted(abs_mu.items(), key=lambda item: item[1][1], reverse=True)}

    return abs_mu


def get_struct_distribution(model, pdb):
    """
    Encodes a PDB structure with CoordVAE and 
    returns its latent mean (mu) and log-variance (logvar).
    """
    inp_struct = np.array([CoordVAE.preprocess_coords(pdb)])
    inp_struct = torch.tensor(inp_struct, dtype=torch.float32)
    
    with torch.no_grad():
        mu, logvar = model.ts_model.encode(inp_struct)
    
    return mu, logvar


def plot_normal_distributions(mu0, logvar0, mu1, logvar1, x_range=(-5, 5), num_points=1000):
    """
    Plots Gaussian distributions for each latent dimension of two structural 
    embeddings by computing probability density curves from their means and 
    variances, arranging them in a grid of subplots for visual comparison.
    """
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    fig, axes = plt.subplots(4, 8, figsize=(10, 7))
    axes = axes.flatten()
    
    for i in range(32):
        mean0 = mu0[0, i].item()
        logvar0_value = logvar0[0, i].item()
        sigma0 = torch.exp(0.5 * torch.tensor(logvar0_value))
        y_values0 = (1 / (sigma0 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean0) / sigma0)**2)

        mean1 = mu1[0, i].item()
        logvar1_value = logvar1[0, i].item()
        sigma1 = torch.exp(0.5 * torch.tensor(logvar1_value))
        y_values1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean1) / sigma1)**2)
        
        axes[i].plot(x_values, y_values0, label=f"mu0_{i+1}")
        axes[i].plot(x_values, y_values1, label=f"mu1_{i+1}")
        
        axes[i].set_title(f"Feature {i+1}")
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()


def produce_specific_aa(keys_values, num_samples=10, temperature=1):
    """
    Generates amino acid sequences by fixing specific latent dimensions to 
    given values, decoding multiple samples at a chosen temperature, 
    and returning the results as a dataframe.
    """
    z1 = torch.zeros(1, model.latent_dim).to(device)
    z2 = torch.zeros(1, model.latent_dim).to(device)

    for key, value in keys_values.items():
        z2[0, key] = value

    print(z2)
    
    samples = []
    for i in range(num_samples):
        out = check_structure(model, None, z_s=[z1, z2], temperature=temperature)
        samples.append(out)
    
    return pd.DataFrame(samples, columns=[["Sequence"]])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension in VAEs")
    args = parser.parse_args()

    print("Loading data...")
    DATA_PATH = "./Data/3LC/CPP_3LC_FT.csv"
    df = pd.read_csv(DATA_PATH, header=0)

    print("Loading Model...")
    model = setup_model(args.latent_dim)
    
    res = compare_structure_pairs(model, df)
    plot_with_fit(res)
    
    """
    compare_distributions(model, df, 0, 560)
    """
    
    """
    spec_dict = rank_dist(model, df, 0)
    df_out = produce_specific_aa(spec_dict, num_samples=1000, temperature=1)
    df_out.to_csv("./Data/Eval/Manual_Latent.csv", index=False)
    """
    
    """
    # spiral: 0, string: 560
    df_out, filename = generate_similar_aas(model, df, row_idx=560, amount=1000, temperature=1, random_sample=False)
    df_out.to_csv("./Data/Eval/{}.csv".format(filename), index=False)
    """

