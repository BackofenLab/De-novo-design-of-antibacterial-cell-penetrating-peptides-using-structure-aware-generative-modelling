import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from globals import tokenizer, device


def apply_rope(x: torch.Tensor):
    """
    Applies RoPE to a tensor of shape (B, H, L, d).
    """
    assert x.ndim == 4, "Expected 4D tensor of shape (B, H, L, d)"
    B, H, L, d = x.shape
    assert d % 2 == 0, "Head dimension must be even for RoPE."

    device = x.device
    half_d = d // 2

    freq = torch.exp(
        -torch.arange(0, half_d, dtype=torch.float32, device=device) *
        (math.log(10000.0) / half_d)
    )  # (half_d,)
    pos = torch.arange(L, dtype=torch.float32, device=device)  # (L,)
    sinusoid = torch.outer(pos, freq)  # (L, half_d)

    sin = sinusoid.sin()[None, None, :, :]  # (1, 1, L, half_d)
    cos = sinusoid.cos()[None, None, :, :]  # (1, 1, L, half_d)

    x1 = x[..., 0::2]  # (B, H, L, half_d)
    x2 = x[..., 1::2]

    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rotated.flatten(-2)


class RotarySelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask=None, causal_mask=None):
        B, S, E = x.shape

        qkv = self.qkv_proj(x).view(B, S, self.num_heads, 3 * self.head_dim).transpose(1, 2)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = apply_rope(Q)
        K = apply_rope(K)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1) 
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, S, E)
        out = self.o_proj(out)

        return out, attn_weights


class RotaryTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.rotary_attn = RotarySelfAttention(embedding_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.gamma1 = nn.Parameter(torch.tensor(1.0))
        self.gamma2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, pad_mask=None, causal_mask=None):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.rotary_attn(x_norm, pad_mask, causal_mask)
        x = x + self.gamma1 * attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.gamma2 * ffn_out

        return x, attn_weights


class PeptideEmbedNet(nn.Module):
    def __init__(self,
                 vocab_size=tokenizer.vocab_size,
                 embedding_dim:int = 512,
                 context_window:int = 512,
                 num_heads:int = 8 ,
                 num_layers:int = 4,
                 ffn_dim:int = 1024,
                 causal_masking:bool = True):
        super(PeptideEmbedNet, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.causal_masking = causal_masking

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.rot_trans_layers = nn.ModuleList([
            RotaryTransformer(embedding_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, encoded: torch.Tensor):
        input_ids = encoded["input_ids"].to(device)
        embeddings = self.token_embeddings(input_ids)

        padding_mask = (encoded["attention_mask"] == 0).to(device)

        if self.causal_masking:
            seq_len = input_ids.size(1)
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
                diagonal=1
            )
        else:
            causal_mask = None

        hidden_1 = embeddings
        attention_maps = []
        for layer in self.rot_trans_layers:
            hidden_1, attn_weights = layer(hidden_1, padding_mask, causal_mask)
            attention_maps.append(attn_weights)

        out = self.linear(hidden_1)
        details = {
                    "embeddings": hidden_1,
                    "attention_maps": attention_maps
                }
        return out, details
