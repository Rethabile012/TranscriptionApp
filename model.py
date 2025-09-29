import torch
import torch.nn as nn

class SpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None, num_layers=3, dropout=0.2):
        super(SpeechModel, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully connected layer maps hidden states to vocab size (+ blank)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x, lengths=None):
        if lengths is not None:
            # Pack padded sequence for efficiency
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(x)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        # Project to vocab dimension
        out = self.fc(out)  # (batch, time, vocab_size)
        return out  # raw logits (no softmax here)
