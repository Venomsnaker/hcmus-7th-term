import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        B, T, C = query.size()

        Q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1,2)  # (B, heads, T, head_dim)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)   # (B, heads, S, head_dim)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1,2) # (B, heads, S, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)         # (B, heads, T, S)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=src_mask)
        x = residual + self.dropout(x)

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src_tokens, src_mask=None):
        B, S = src_tokens.size()
        positions = torch.arange(0, S, device=src_tokens.device).unsqueeze(0).expand(B, S)

        x = self.token_embedding(src_tokens) + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        x = self.norm(x)
        return x


# Example usage:
if __name__ == "__main__":
    batch_size = 2
    src_len = 15
    vocab_size = 1000
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6

    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, vocab_size)
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_len))

    encoder_outputs = encoder(src_tokens)
    print(encoder_outputs.shape)  # (batch_size, src_len, embed_dim)
