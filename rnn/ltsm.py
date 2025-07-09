import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMScratchCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate parameters
        self.W_xi = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        
        # Forget gate parameters
        self.W_xf = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        
        # Output gate parameters
        self.W_xo = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
        # Cell gate parameters (also called cell candidate)
        self.W_xc = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hc = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, states):
        # x: input at current time step, shape (batch, input_size)
        # states: tuple of (hidden_state, cell_state), each shape (batch, hidden_size)
        h_prev, c_prev = states
        
        i = torch.sigmoid(x @ self.W_xi + h_prev @ self.W_hi + self.b_i)    # input gate
        f = torch.sigmoid(x @ self.W_xf + h_prev @ self.W_hf + self.b_f)    # forget gate
        o = torch.sigmoid(x @ self.W_xo + h_prev @ self.W_ho + self.b_o)    # output gate
        c_tilde = torch.tanh(x @ self.W_xc + h_prev @ self.W_hc + self.b_c) # candidate cell
        
        c = f * c_prev + i * c_tilde  # update cell state
        h = o * torch.tanh(c)         # update hidden state
        
        return h, c
