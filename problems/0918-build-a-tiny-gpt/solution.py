
import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()

        # 1. Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # 2. Positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                dropout=0.0
            )
            for _ in range(num_layers)
        ])

        # 4. Final Layer Normalization
        self.ln_f = nn.LayerNorm(d_model)

        # 5. Output projection
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        # Token embeddings
        token_embeddings = self.token_emb(idx)

        # Positional embeddings
        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.pos_emb(positions)

        # Combine token and positional embeddings
        x = token_embeddings + position_embeddings

        # Causal mask: prevent attending to future tokens
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask)

        # Final normalization
        x = self.ln_f(x)

        # Project to vocabulary logits
        logits = self.head(x)

        return logits
