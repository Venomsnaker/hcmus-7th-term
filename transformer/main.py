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
        # query/key/value shape: (batch_size, seq_len, embed_dim)
        B, T, C = query.size()

        # Linear projections
        Q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)   # (B, heads, S, head_dim)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, S, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, S)

        if attn_mask is not None:
            # attn_mask shape should be broadcastable to (B, heads, T, S)
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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, tgt_mask=None, memory_mask=None):
        # x: (B, T, C)
        # encoder_outputs: (B, S, C)
        # tgt_mask: mask for self-attention (causal mask)
        # memory_mask: mask for cross-attention (optional)

        # Self-attention with causal mask
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = residual + self.dropout(x)

        # Cross-attention over encoder outputs
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, encoder_outputs, encoder_outputs, attn_mask=memory_mask)
        x = residual + self.dropout(x)

        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + self.dropout(x)

        return x

def generate_causal_mask(sz):
    # Generate a causal mask for self-attention to prevent attending to future tokens
    mask = torch.tril(torch.ones(sz, sz)).unsqueeze(0).unsqueeze(0)  # (1,1,sz,sz)
    return mask  # 1 means keep, 0 means mask out

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_tokens, encoder_outputs, tgt_mask=None, memory_mask=None):
        # tgt_tokens: (B, T)
        B, T = tgt_tokens.size()
        positions = torch.arange(0, T, device=tgt_tokens.device).unsqueeze(0).expand(B, T)

        x = self.token_embedding(tgt_tokens) + self.pos_embedding(positions)

        if tgt_mask is None:
            tgt_mask = generate_causal_mask(T).to(tgt_tokens.device)

        for layer in self.layers:
            x = layer(x, encoder_outputs, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.norm(x)
        logits = self.output_proj(x)  # (B, T, vocab_size)
        return logits

# Example usage:
if __name__ == "__main__":
    batch_size = 2
    tgt_len = 10
    src_len = 15
    vocab_size = 1000
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6

    decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, vocab_size)
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))
    encoder_outputs = torch.randn(batch_size, src_len, embed_dim)

    logits = decoder(tgt_tokens, encoder_outputs)
    print(logits.shape)  # (batch_size, tgt_len, vocab_size)
