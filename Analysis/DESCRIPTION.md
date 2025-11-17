### Analysis/DATASET_HISTOGRAM_FIGURE.py  
Reads peptide sequence data from `Data/classified_structures.csv`, computes sequence length and net charge at pH 7.0 using `Bio.SeqUtils.ProtParam`, and visualizes the distributions of sequence lengths, charges, and structural classes using `seaborn` and `matplotlib`.  
**Accesses:** `Data/classified_structures.csv`  
**Outputs:** `DATASET_HISTOGRAM_FIGURE.png`

---

### Analysis/DATASET_PREP_FIGURE.py  
Generates a publication-quality flowchart (PNG format) using `Graphviz` to illustrate the dataset preprocessing pipeline for Supplementary Figure S1b. It shows dataset merging (CPPSite 2.0, CellPPD), duplicate removal, structure prediction via `ESMFold`, and creation of the validated CPP dataset.  
**Accesses:** none  
**Outputs:** `dataset_pipeline_vertical.png`

---

### Analysis/ESM_analysis.py  
Clusters protein tertiary structures (PDB strings) using RMSD-based agglomerative clustering. Computes RMSD matrices from alpha carbon (CA) atoms, generates 3D line-plot images, and saves clustering results as HTML.  
**Accesses:** `Data/CPP_AAS_with_structures.csv`  
**Outputs:** `Data/clustered_peptides.html`  
*(Note: clustering was later replaced by manual classification in `classify_cpps.py`.)*

---

### Analysis/aa_distribution.py  
Analyzes and visualizes amino acid distributions across multiple peptide datasets. Reads peptide sequences, computes per-residue frequencies excluding ambiguous characters (B, X, Z), and creates bar charts using `matplotlib`.  
**Accesses:**  
- `../Data/CPP_AAS.csv`  
- `../Data/Eval/VAE_FT_EVAL.csv`  
- `../Data/Eval/Similar_To_AAAAARRRIRKQAHAHSK_EVAL.csv`  
- `../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv`  
**Dependencies:** `pandas`, `numpy`, `matplotlib`, `collections`

---

### Analysis/attn.py  
Provides visualization tools for analyzing attention mechanisms in transformer-based models. Extracts attention maps from the model and dataloader, tokenizes inputs via `globals.py`, and visualizes attention per layer.  
**Accesses:** model attention outputs, tokenizer from `globals.py`  
**Outputs:** static (matplotlib) and interactive (Plotly) attention plots

---

### Analysis/classify_cpps.py  
Implements a `Streamlit` app for classifying peptide tertiary structures into Spiral, String, and NA. Loads structures, displays them using `render_pdb_to_image` from `visualize_pdb.py`, and allows manual labeling.  
**Accesses:** `../Data/classified_structures.csv`, `visualize_pdb.py`  
**Outputs:** updated `../Data/classified_structures.csv` with user-assigned labels

---

### Analysis/cpp_lengths.py  
Analyzes and visualizes peptide length distributions across datasets. Computes sequence lengths and creates histograms and violin plots using `matplotlib`.  
**Accesses:** `../Data/CPP_AAS.csv`, files in `../Data/Eval/`  
**Outputs:** PNG figures of length distributions

---

### Analysis/data_analysis.py  
Analyzes token usage and entropy in biological sequence data. Builds substitution families and calculates token frequencies and positional entropy, then visualizes results.  
**Accesses:**  
- `../Data/TESTING.csv`  
- `../Data/SSD_Derivatives.txt`  
**Outputs:** frequency bar plots and entropy line plots

---

### Analysis/loss.py  
Parses and visualizes training/test loss data. Extracts epoch, step, and loss values and plots loss curves using `matplotlib`.  
**Accesses:** `./log.txt`  
**Outputs:** loss progression plots

---

### Analysis/loss_plots.py  
Parses training log files and visualizes epoch-wise training/test loss progression. Uses regex parsing and `matplotlib` for plotting.  
**Accesses:** `../Checkpoints/log.txt`  
**Outputs:** loss comparison figure

---

### Analysis/non_nat_clean.py  
Filters sequences with non-natural amino acids. Removes entries with `[UNK]` or invalid residues and saves cleaned results.  
**Accesses:** `../Data/Eval/NON_NAT.csv`  
**Outputs:** `../Data/Eval/NON_NAT_cleaned.csv`

---

### Analysis/pairwise_dist.py  
Computes pairwise sequence alignment distances using `parasail` and the `BLOSUM62` matrix, visualizing results as heatmaps.  
**Accesses:**  
- `Data/CPP_AAS.csv`  
- Files in `Data/Eval/` (e.g., `VAE_FT_EVAL.csv`)  
**Outputs:** heatmap visualizations (PNG)  
**Dependencies:** `pandas`, `numpy`, `parasail`, `matplotlib`, `tqdm`

---

### Analysis/properties.py  
Computes and visualizes physicochemical properties (MW, logP, tPSA, etc.) of peptides. Uses `RDKit` for computation and `matplotlib` for boxplot visualization. Highlights value ranges for known CPPs (de Oliveira et al., 2021).  
**Accesses:** datasets in `../Data/` and `../Data/Eval/`  
**Outputs:** comparative boxplots  
**Dependencies:** `RDKit`, `matplotlib`, `pandas`

---

### Analysis/topk.py  
Finds top-k generated CPPs most similar to validated CPPs using Euclidean distance in physicochemical property space. Computes molecular properties via `RDKit`, normalizes with `scikit-learn`, and visualizes via radar plots.  
**Accesses:**  
- `../Data/CPP_AAS.csv`  
- `../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv`  
**Outputs:** `../Data/Top_k/String_top_30.csv` (optional)  
**Dependencies:** `RDKit`, `scikit-learn`, `matplotlib`

---

### Analysis/visualize_pdb.py  
Predicts and visualizes peptide 3D structures using `ESMFold` (facebook/esmfold_v1) and `PyMOL`. Renders molecular images, overlays sequences using `PIL`, and compiles results.  
**Accesses:** `../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv`  
**Outputs:** combined visualization images  
**Dependencies:** `PyMOL`, `PIL`, `pandas`, `torch`, `transformers`

