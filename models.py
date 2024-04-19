import torch.nn as nn
class Predictor(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Predictor, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, embeddings):
        output, _ = self.lstm(embeddings)
        scores = self.linear(output[:, -1, :])  # Use only the last LSTM output
        return scores