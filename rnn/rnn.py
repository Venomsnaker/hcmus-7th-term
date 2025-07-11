import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, hn = self.rnn(x, h0)
        
        # out: batch_size, seq_length, hidden_size
        # Use last time step's output for prediction
        out = out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(out)
        return out

# Example usage
if __name__ == "__main__":
    batch_size = 16
    seq_length = 10
    input_size = 8
    hidden_size = 32
    output_size = 2  # e.g., classification with 2 classes

    # Create random input with shape (batch_size, seq_length, input_size)
    x = torch.randn(batch_size, seq_length, input_size)

    model = SimpleRNN(input_size, hidden_size, output_size)
    output = model(x)
    print(output.shape)  # Expected: (batch_size, output_size)
