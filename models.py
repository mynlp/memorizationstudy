import torch.nn as nn
class Predictor(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers=2, drop_prob=0.5):
        super(Predictor, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, embeddings):
        output, _ = self.lstm(embeddings)
        output = self.dropout(output[:, -1, :])
        output = self.linear1(output)
        output = self.relu(output)
        scores = self.linear2(output)
        return scores
