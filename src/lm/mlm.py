import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmMlm(nn.Module):
    def __init__(self, num_tokens, embedding_dim=512, model_dim=784, n_layers=2):
        super().__init__()
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.up_proj = nn.Linear(embedding_dim, model_dim)
        self.n_layers = n_layers
        encoders = []
        for _ in range(n_layers):
            encoder = nn.LSTM(model_dim, int(model_dim / 2), num_layers=1, bidirectional=True, batch_first=True)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        self.down_proj = nn.Linear(model_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_tokens)


    def forward(self, input, input_len):
        """
        input: (batch_size x seq_len)
        """
        embedded = self.embedding(input) # (batch_size, seq_len, model_dim)
        embedded = self.up_proj(embedded)
        embedded_len = input_len
        hidden_states = []
        hidden = None
        for encoder in self.encoders:
            # Create packed sequence
            embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, embedded_len.cpu().numpy(), batch_first=True, enforce_sorted=False)

            # Compute hidden states
            hidden, _ = encoder(embedded_packed)

            # Create padded sequence
            hidden, embedded_len = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)
            # Add to hidden_states
            hidden_states.append(hidden)

            # Layer normalization
            forward_hidden = hidden[:, :, :self.model_dim]
            backward_hidden = hidden[:, :, self.model_dim:]
            forward_hidden = F.layer_norm(forward_hidden, forward_hidden.size()[1:])
            backward_hidden = F.layer_norm(backward_hidden, backward_hidden.size()[1:])

            # Residual connection
            hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
            embedded = embedded + hidden

        output = self.down_proj(hidden)
        output = self.fc(output)
        return output, torch.stack(hidden_states)
