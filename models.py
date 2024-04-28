import torch.nn as nn
import torch
import pdb
class Predictor(nn.Module):
    def __init__(self, embedding_size, hidden_size, context_size=32,  num_layers=2, drop_prob=0.5):
        super(Predictor, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
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
        pdb.set_trace()
        classes = torch.nn.LogSoftmax(classes)  # if you want output in [0, 1]

        return classes
