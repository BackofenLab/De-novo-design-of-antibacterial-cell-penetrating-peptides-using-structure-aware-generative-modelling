"""
Script for producing images for pdb structures using PyMol and PIL.
"""

import os
import pandas as pd
import torch
from transformers import EsmForProteinFolding
from tqdm import tqdm
import tempfile
from pymol import cmd, finish_launching
from PIL import Image, ImageDraw, ImageFont
import random


def load_model(device):
    """
    Loads the pretrained ESMFold model, sets it to evaluation mode, 
    moves it to the specified device, and returns it.
    """
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.eval().to(device)
    return model


def get_pdb(model, seq):
    """
    Generates a predicted PDB structure string for a given protein 
    sequence using the provided model in no-gradient mode.
    """
    with torch.no_grad():
        return model.infer_pdb(seq)


def start_pymol():
    """
    Initializes a PyMOL session in quiet mode, resets the environment, 
    and sets the background color to white.
    """
    finish_launching(['pymol', '-cq'])
    cmd.reinitialize()
    cmd.bg_color("white")


def render_pdb_to_image(aa_seq: str, pdb_string: str, name: str = "molecule") -> Image.Image:
    """
    Renders a PDB structure into a high-resolution PyMOL image, 
    overlays the amino acid sequence as a text label, and returns the 
    result as a PIL image object.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
        tmp_pdb.write(pdb_string.encode("utf-8"))
        pdb_path = tmp_pdb.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        img_path = tmp_img.name

    cmd.load(pdb_path, name)
    cmd.show("cartoon", name)
    cmd.zoom(name)
    cmd.png(img_path, width=800, height=600, dpi=300, ray=1)
    cmd.delete(name)
    os.remove(pdb_path)

    with Image.open(img_path) as img_file:
        img = img_file.copy()

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=30)
    text_position = (10, 10)
    draw.text(text_position, aa_seq, font=font, fill=(0, 0, 0))

    os.remove(img_path)
    return img


def combine_images(images, cols=5, padding=10, bg_color=(255, 255, 255)):
    """
    Combines a list of images into a grid with configurable columns, 
    padding, and background color, and returns the merged image.
    """
    if not images:
        return None

    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    result = Image.new("RGB", (cols * w + (cols - 1) * padding,
                               rows * h + (rows - 1) * padding), bg_color)

    for idx, img in enumerate(images):
        x = (idx % cols) * (w + padding)
        y = (idx // cols) * (h + padding)
        result.paste(img, (x, y))

    return result


def random_sampling(path, aa_seq=None):
    """
    Performs random sampling of protein sequences from a dataset, 
    generates predicted PDB structures and images for them 
    (optionally including a given input sequence), arranges the 
    results into a combined grid image, and saves both input and
    output images.
    """
    df = pd.read_csv(path)["Sequence"].tolist()
    
    if aa_seq is not None:
        input_struct = aa_seq
        pdb = get_pdb(model, input_struct)
        input_img = render_pdb_to_image(aa_seq, pdb)
        input_img.save("Input.png")

    imgs = []
    for seq in tqdm(random.sample(df, 25), desc="Processing sequences"):
        pdb = get_pdb(model, seq)
        img = render_pdb_to_image(seq, pdb)
        imgs.append(img)

    cmd.quit()
    final_img = combine_images(imgs)
    final_img.save("Output.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    start_pymol()
    
    random_sampling("../Data/Eval/Similar_To_KMDRWRWKKK_EVAL.csv", "KMDRWRWKKK")


