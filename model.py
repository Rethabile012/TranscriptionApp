
import torch.nn as nn

class SpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None, num_layers=3, dropout=0.2):
        super(SpeechModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0  # PyTorch applies dropout only if num_layers > 1
        )
        # Fully connected layer maps hidden states to output (vocab size)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, time, hidden*2)
        out = self.fc(out)           # (batch, time, vocab_size)
        return self.softmax(out)     # log probabilities for CTC
