"""
Experimental script for analysing the attention in transformer based models.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from globals import tokenizer


def plot_attn_heatmap(attn_map, tokens):
    """
    Generates and saves a heatmap of token-to-token attention weights using matplotlib, 
    labeling tokens on both axes and coloring by attention strength.
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(attn_map, cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title("Token-to-Token Attention Map")

    token_num = len(tokens)
    step = 1
    plt.xticks(range(0, token_num, step), tokens[0:token_num:step], rotation=90)
    plt.yticks(range(0, token_num, step), tokens[0:token_num:step])

    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    

def plot_attn_lines(attn_layer, tokens, step=1, row_spacing=3, min_weight=0.01):
    """
    Creates an interactive Plotly visualization of token-to-token attention 
    across all heads, drawing weighted lines between tokens and stacking heads vertically, 
    then saves the result as an HTML file.
    """
    idx = np.arange(0, len(tokens), step)
    tokens = [tokens[i] for i in idx]
    x = np.linspace(0, 1, len(tokens))

    fig = go.Figure()
    num_heads = attn_layer.shape[0]

    for h in tqdm(range(num_heads)):
        attn = attn_layer[h][idx][:, idx]
        
        y_top = row_spacing * (num_heads - h)
        y_bottom = y_top - 2.5

        # Add attention lines
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                w = attn[i, j]
                if w > min_weight:
                    fig.add_trace(go.Scatter(
                        x=[x[i], x[j]],
                        y=[y_top-0.2, y_bottom+0.2],
                        mode='lines',
                        line=dict(width=0.5, color=f'rgba(0,0,255,{w:.3f})'),
                        hoverinfo='none',
                        showlegend=False
                    )) 

        # Add token labels
        fig.add_trace(go.Scatter(
            x=x,
            y=[y_top] * len(tokens),
            mode='text',
            text=tokens,
            textposition="bottom center",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=[y_bottom] * len(tokens),
            mode='text',
            text=tokens,
            textposition="top center",
            showlegend=False
        ))

        # Add head label
        fig.add_trace(go.Scatter(
            x=[-0.05],
            y=[(y_top + y_bottom) / 2],
            mode='text',
            text=[f'Head {h + 1}'],
            textposition="middle right",
            showlegend=False
        ))

    fig.update_layout(
        title="Token-to-Token Attention for All Heads",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
        height=row_spacing * num_heads * 100,
        width=10000
    )

    fig.write_html("attention_all_heads.html")


def analyse_attn(model, dataloader, transformer_layer=0):
    """
    Extracts and visualizes attention from a specified transformer layer by running 
    the model on a sample from the dataloader, tokenizing the input, retrieving 
    attention maps, and plotting token-to-token connections with plot_attn_lines.
    """
    model.eval()
    datapoint = next(iter(dataloader))[0]

    tokenized_seq = tokenizer(datapoint,
                              padding="max_length",
                              max_length=model.context_window,
                              truncation=True,
                              return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(tokenized_seq["input_ids"][0].tolist())

    out, details = model(tokenized_seq)
    attn = details["attention_maps"]  # list of Batch x Head x context_window x context_window

    # attn = [t.mean(dim=1).detach().numpy() for t in attn]  # average over heads
    

    plot_attn_lines(attn[transformer_layer][0].detach().cpu().numpy(), tokens)
