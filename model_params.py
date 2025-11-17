"""
Script for computing the amount of parameters of all models.
"""

import numpy as np
import pandas as pd
import torch

from Model.TSC_vae import CoordVAE
from Model.model import PeptideEmbedNet
from Model.vae import TransformerVAE
from globals import device, tokenizer


def load_models(cvae_path, pen_path, tvae_path):
    """
    Loads pretrained weights for CoordVAE, PeptideEmbedNet, and 
    TransformerVAE from checkpoint files, initializes the models 
    with the correct configurations, and returns all three.
    """
    coordvae = CoordVAE(latent_dim=32)
    state_dict = torch.load(cvae_path, map_location=torch.device("cpu"))
    coordvae.load_state_dict(state_dict)

    pt_model = PeptideEmbedNet(num_layers=1,
                               ffn_dim=512,
                               context_window=64,
                               causal_masking=False)
    state_dict = torch.load(pen_path, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict)


    model = TransformerVAE(
        pretrained_embedding_model=pt_model,
        pretrained_structure_model=coordvae,
        latent_dim=32,
        vocab_size=tokenizer.vocab_size,
        num_heads=4,
        max_len=100
    )
    state_dict = torch.load(tvae_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    return coordvae, pt_model, model


def count_params(model):
    """
    Calculates the total and trainable parameter counts of a model, 
    excluding parameters from frozen pretrained components, and 
    returns both values.
    """
    overall_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith(("pt_model.", "ts_model."))  # Those are frozen during training
    )
    return overall_params, trainable_params
    

if __name__ == "__main__":
    cvae_path = "./Checkpoints/2025_07_05_TS/ts_vae.pt"
    pen_path = "./Checkpoints/2025_07_03_MLM/model_20250703_134624.pth"
    tvae_path = "./Checkpoints/2025_07_05_VAE/vae.pth"

    cvae, pen, tvae = load_models(cvae_path, pen_path, tvae_path)


    print(count_params(cvae))
    print(count_params(pen))
    print(count_params(tvae))


