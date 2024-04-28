import torch.nn as nn
import torch
import pdb
class LSTMPredictor(nn.Module):
    def __init__(self, embedding_size, hidden_size, context_size=32,  num_layers=3, drop_prob=0.5):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_prob)
        self.context_size = context_size
        self.dropout = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size+1, 2)
    def forward(self, embeddings, entropy):
        output, _ = self.lstm(embeddings)
        #output = self.dropout(output)
        output = self.linear1(output)
        output = self.relu(output)
        selected_output = output[:, self.context_size-1:, :]
        selected_output = torch.cat((selected_output, entropy.unsqueeze(2)), dim=2)
        classes = self.linear3(selected_output)  # newly added for classes output
        #pdb.set_trace()
        classes = torch.nn.functional.log_softmax(classes, dim=1)  # if you want output in [0, 1]

        return classes

class TransformerPredictor(nn.Module):
    def __init__(self, embedding_size, hidden_size, context_size=32, num_layers=3, drop_prob=0.5):
        super(TransformerPredictor, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.context_size = context_size
        self.dropout = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size + 1, 2)

    def forward(self, embeddings, entropy):
        output = self.transformer(embeddings)
        output = self.linear1(output)
        output = self.relu(output)
        selected_output = output[:, self.context_size - 1:, :]
        selected_output = torch.cat((selected_output, entropy.unsqueeze(2)), dim=2)
        classes = self.linear3(selected_output)
        classes = torch.nn.functional.log_softmax(classes, dim=1)

        return classes