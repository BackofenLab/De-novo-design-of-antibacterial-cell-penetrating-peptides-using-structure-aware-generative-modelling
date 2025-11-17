import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.model import PeptideEmbedNet


class TransformerVAE(nn.Module):
    def __init__(self,
                 pretrained_embedding_model,
                 pretrained_structure_model,
                 latent_dim=256,
                 vocab_size=33,
                 max_len=100,
                 num_heads=8):
        super().__init__()
        self.pt_model = pretrained_embedding_model
        self.ts_model = pretrained_structure_model
        self.embed_dim = self.pt_model.embedding_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.pos_embed = nn.Parameter(torch.randn(1, max_len, self.embed_dim))

        # Encoder
        self.encoder_mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        self.encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_fc_mu = nn.Linear(self.embed_dim, latent_dim)
        self.encoder_fc_logvar = nn.Linear(self.embed_dim, latent_dim)

        # Decoder
        self.latent_to_decoder = nn.Linear(latent_dim, self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        self.output_fc = nn.Linear(self.embed_dim, vocab_size)

    def encode(self, x):
        x = x + self.pos_embed[:, :x.size(1), :]
        attn_out, _ = self.encoder_mha(x, x, x)
        x = self.encoder_norm(x + attn_out)
        pooled = x.mean(dim=1)
        mu = self.encoder_fc_mu(pooled)
        logvar = self.encoder_fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z1, z2, tgt_len):
        q = self.latent_to_decoder(z1).unsqueeze(1).repeat(1, tgt_len, 1)
        kv = self.latent_to_decoder(z2).unsqueeze(1).repeat(1, tgt_len, 1)

        tgt = q + self.pos_embed[:, :tgt_len, :]
        attn_out, _ = self.cross_attn(tgt, kv, kv)
        x = self.decoder_norm(tgt + attn_out)
        return self.output_fc(x)

    def forward(self, x, structure_x):
        with torch.no_grad():
            _, x = self.pt_model(x)
            x = x["embeddings"]

        mu, logvar = self.encode(x)
        z1 = self.reparameterize(mu, logvar)
        
        with torch.no_grad():
            mu2, logvar2 = self.ts_model.encode(structure_x)
            z2 = self.ts_model.reparameterize(mu2, logvar2)

        logits = self.decode(z1, z2, tgt_len=x.size(1))
        return logits, mu, logvar
