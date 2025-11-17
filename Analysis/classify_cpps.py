"""
Script for classifying the tertiary structures of peptides into Spiral, String and NA.
"""

import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import os

from visualize_pdb import render_pdb_to_image


LABEL_OPTIONS = ["Spiral", "String", "NA"]
LABEL_COL = "label"
SAVE_PATH = "../Data/classified_structures.csv"
SOURCE_PATH = SAVE_PATH # "../Data/CPP_AAS_with_structures.csv"


def load_or_initialize_data():
    """
    Loads data from a saved file if available to resume progress; otherwise loads the source file, 
    initializes labels as "UNLABELED", and returns the dataframe.
    """
    if os.path.exists(SAVE_PATH):
        df = pd.read_csv(SAVE_PATH)
        st.info("Resumed from previous progress.")
    else:
        df = pd.read_csv(SOURCE_PATH)
        df[LABEL_COL] = "UNLABELED"
        st.info("Started fresh.")
    return df


def classify_structure(df):
    """
    Displays a protein structure for the current index, allows the user to assign 
    a label via Streamlit, saves the selection, advances to the next unlabeled entry, 
    and updates the dataframe.
    """
    idx = st.session_state.get("current_index", 0)
    aa_seq = df.loc[idx, "AA"]
    pdb_str = df.loc[idx, "pdb_structure"]
    image = render_pdb_to_image(aa_seq, pdb_str)
    st.image(image, caption=f"Index {idx} | AA: {aa_seq}", use_column_width=True)

    label = st.radio("Select label:", LABEL_OPTIONS, key=f"label_{idx}")
    if st.button("Save label", key=f"save_{idx}"):
        df.at[idx, LABEL_COL] = label
        df.to_csv(SAVE_PATH, index=False)
        next_idx = df[df[LABEL_COL] == "UNLABELED"].index
        if len(next_idx) > 0:
            st.session_state.current_index = next_idx[0]
            st.rerun()
        else:
            st.success("All structures labeled!")
    return df


def main():
    """
    Runs the main Streamlit app for classifying PDB structures, managing labeling 
    progress, displaying the current structure for annotation, and providing an option 
    to download the updated dataset.
    """
    st.title("PDB Structure Classifier")

    df = load_or_initialize_data()

    if df[LABEL_COL].eq("UNLABELED").sum() == 0:
        st.success("All structures have been labeled!")
        st.dataframe(df)
        return

    if "current_index" not in st.session_state:
        st.session_state.current_index = df[df[LABEL_COL] == "UNLABELED"].index[0]

    st.markdown(f"**Current index: {st.session_state.current_index}**")
    df = classify_structure(df)

    st.markdown("---")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="classified_structures.csv")


def validate():
    """
    Provides a Streamlit interface to review labeled protein structures, 
    filter them by label, view structures with their metadata, and download 
    the complete labeled dataset as a CSV file.
    """
    st.title("Labeled PDB Structure Viewer")

    df = load_or_initialize_data()
    labeled_df = df[df[LABEL_COL] != "UNLABELED"]

    if labeled_df.empty:
        st.warning("No labeled structures yet.")
        return

    st.markdown(f"**{len(labeled_df)} labeled structures found.**")

    selected_label = st.selectbox("Filter by label:", ["All"] + LABEL_OPTIONS)
    if selected_label != "All":
        labeled_df = labeled_df[labeled_df[LABEL_COL] == selected_label]

    idx = st.selectbox("Select index to view", labeled_df.index.tolist())
    aa_seq = labeled_df.loc[idx, "AA"]
    pdb_str = labeled_df.loc[idx, "pdb_structure"]
    label = labeled_df.loc[idx, LABEL_COL]

    image = render_pdb_to_image(aa_seq, pdb_str)
    st.image(image, caption=f"Index {idx} | Label: {label} | AA: {aa_seq}", use_column_width=True)

    st.download_button("Download CSV", df.to_csv(index=False), file_name="classified_structures.csv")


if __name__ == "__main__":
    validate()


