import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()

        self.rotation_matrix = torch.zeros(d_model, d_model, device=torch.device("cuda"))
        self.positional_embedding = torch.zeros(max_seq_len, d_model, device=torch.device("cuda"))

        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)

        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(i * j * 0.01)

    def forward(self, x):
        x += self.positional_embedding
        x = torch.matmul(x, self.rotation_matrix)
        return x