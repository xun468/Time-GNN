import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, loss, **kwargs):
        super(LSTM, self).__init__()
        
        self.loss = loss 
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        ).to(device)

        # Fully connected layer[[[]]]
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

        # self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal(self.lstm.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal(self.fc.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # print(x.shape)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        out = self.fc(out)

        return out