import pandas as pd
energy_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')

energy_df['date'] = pd.to_datetime(energy_df['date'])

energy_df['month'] = energy_df['date'].dt.month.astype(int)
energy_df['day_of_month'] = energy_df['date'].dt.day.astype(int)

# day_of_week=0 corresponds to Monday
energy_df['day_of_week'] = energy_df['date'].dt.dayofweek.astype(int)
energy_df['hour_of_day'] = energy_df['date'].dt.hour.astype(int)

selected_columns = ['date', 'day_of_week', 'hour_of_day', 'Appliances']
energy_df = energy_df[selected_columns]
import numpy as np

resample_df = energy_df.set_index('date').resample('1H').mean()
resample_df['date'] = resample_df.index
resample_df['log_energy_consumption'] = np.log(resample_df['Appliances'])

datetime_columns = ['date', 'day_of_week', 'hour_of_day']
target_column = 'log_energy_consumption'

feature_columns = datetime_columns + ['log_energy_consumption']

# For clarity in visualization and presentation,
# only consider the first 150 hours of data.
resample_df = resample_df[feature_columns]
import plotly.express as px

plot_length = 150
plot_df = resample_df.copy(deep=True).iloc[:plot_length]
plot_df['weekday'] = plot_df['date'].dt.day_name()

fig = px.line(plot_df,
              x="date",
              y="log_energy_consumption",
              color="weekday",
              title="Log of Appliance Energy Consumption vs Time")
fig.show()
from sklearn.preprocessing import MinMaxScaler

def create_sliding_window(data, sequence_length, stride=1):
    X_list, y_list = [], []
    for i in range(len(data)):
      if (i + sequence_length) < len(data):
        X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
        y_list.append(data.iloc[i+sequence_length, -1])
    return np.array(X_list), np.array(y_list)

train_split = 0.7
n_train = int(train_split * len(resample_df))
n_test = len(resample_df) - n_train

features = ['day_of_week', 'hour_of_day', 'log_energy_consumption']
feature_array = resample_df[features].values

# Fit Scaler only on Training features
feature_scaler = MinMaxScaler()
feature_scaler.fit(feature_array[:n_train])
# Fit Scaler only on Training target values
target_scaler = MinMaxScaler()
target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))

# Transfom on both Training and Test data
scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                            columns=features)

sequence_length = 10
X, y = create_sliding_window(scaled_array,
                             sequence_length)

X_train = X[:n_train]
y_train = y[:n_train]

X_test = X[n_train:]
y_test = y[n_train:]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):
        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size  # user-defined

        self.hidden_size_1 = 128  # number of encoder cells (from paper)
        self.hidden_size_2 = 32  # number of decoder cells (from paper)
        self.stacked_layers = 2  # number of (stacked) LSTM layers for each stage
        self.dropout_probability = 0.5  # arbitrary value (the paper suggests that performance is generally stable across all ranges)

        self.lstm1 = nn.LSTM(n_features,
                             self.hidden_size_1,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)

        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = output[:, -1, :]  # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred

    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state

    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state

    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()

n_features = scaled_array.shape[-1]
sequence_length = 10
output_length = 1

batch_size = 128
n_epochs = 10
learning_rate = 0.01

bayesian_lstm = BayesianLSTM(n_features=n_features,
                             output_length=output_length,
                             batch_size = batch_size)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=learning_rate)

bayesian_lstm.train()

for e in range(1, n_epochs+1):
    for b in range(0, len(X_train), batch_size):
        features = X_train[b:b+batch_size,:,:]
        target = y_train[b:b+batch_size]

        X_batch = torch.tensor(features,dtype=torch.float32)
        y_batch = torch.tensor(target,dtype=torch.float32)

        output = bayesian_lstm(X_batch)
        loss = criterion(output.view(-1), y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if e % 10 == 0:
      print('epoch', e, 'loss: ', loss.item())
offset = sequence_length

def inverse_transform(y):
  return target_scaler.inverse_transform(y.reshape(-1, 1))

training_df = pd.DataFrame()
training_df['date'] = resample_df['date'].iloc[offset:n_train + offset:1]
training_predictions = bayesian_lstm.predict(X_train)
training_df['log_energy_consumption'] = inverse_transform(training_predictions)
training_df['source'] = 'Training Prediction'

training_truth_df = pd.DataFrame()
training_truth_df['date'] = training_df['date']
training_truth_df['log_energy_consumption'] = resample_df['log_energy_consumption'].iloc[offset:n_train + offset:1]
training_truth_df['source'] = 'True Values'

testing_df = pd.DataFrame()
testing_df['date'] = resample_df['date'].iloc[n_train + offset::1]
testing_predictions = bayesian_lstm.predict(X_test)
testing_df['log_energy_consumption'] = inverse_transform(testing_predictions)
testing_df['source'] = 'Test Prediction'

testing_truth_df = pd.DataFrame()
testing_truth_df['date'] = testing_df['date']
testing_truth_df['log_energy_consumption'] = resample_df['log_energy_consumption'].iloc[n_train + offset::1]
testing_truth_df['source'] = 'True Values'

evaluation = pd.concat([training_df,
                        testing_df,
                        training_truth_df,
                        testing_truth_df
                        ], axis=0)
fig = px.line(evaluation.loc[evaluation['date'].between('2016-04-14', '2016-04-23')],
                 x="date",
                 y="log_energy_consumption",
                 color="source",
                 title="Log of Appliance Energy Consumption in Wh vs Time")
fig.show()
n_experiments = 100

test_uncertainty_df = pd.DataFrame()
test_uncertainty_df['date'] = testing_df['date']

for i in range(n_experiments):
  experiment_predictions = bayesian_lstm.predict(X_test)
  test_uncertainty_df['log_energy_consumption_{}'.format(i)] = inverse_transform(experiment_predictions)

log_energy_consumption_df = test_uncertainty_df.filter(like='log_energy_consumption', axis=1)
test_uncertainty_df['log_energy_consumption_mean'] = log_energy_consumption_df.mean(axis=1)
test_uncertainty_df['log_energy_consumption_std'] = log_energy_consumption_df.std(axis=1)

test_uncertainty_df = test_uncertainty_df[['date', 'log_energy_consumption_mean', 'log_energy_consumption_std']]
test_uncertainty_df['lower_bound'] = test_uncertainty_df['log_energy_consumption_mean'] - 3*test_uncertainty_df['log_energy_consumption_std']
test_uncertainty_df['upper_bound'] = test_uncertainty_df['log_energy_consumption_mean'] + 3*test_uncertainty_df['log_energy_consumption_std']
import plotly.graph_objects as go

test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
truth_uncertainty_plot_df = testing_truth_df.copy(deep=True)
truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

upper_trace = go.Scatter(
    x=test_uncertainty_plot_df['date'],
    y=test_uncertainty_plot_df['upper_bound'],
    mode='lines',
    fill=None,
    name='99% Upper Confidence Bound'
    )
lower_trace = go.Scatter(
    x=test_uncertainty_plot_df['date'],
    y=test_uncertainty_plot_df['lower_bound'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 211, 0, 0.1)',
    name='99% Lower Confidence Bound'
    )
real_trace = go.Scatter(
    x=truth_uncertainty_plot_df['date'],
    y=truth_uncertainty_plot_df['log_energy_consumption'],
    mode='lines',
    fill=None,
    name='Real Values'
    )

data = [upper_trace, lower_trace, real_trace]

fig = go.Figure(data=data)
fig.update_layout(title='Uncertainty Quantification for Energy Consumption Test Data',
                   xaxis_title='Time',
                   yaxis_title='log_energy_consumption (log Wh)')

fig.show()
bounds_df = pd.DataFrame()

# Using 99% confidence bounds
bounds_df['lower_bound'] = test_uncertainty_plot_df['lower_bound']
bounds_df['prediction'] = test_uncertainty_plot_df['log_energy_consumption_mean']
bounds_df['real_value'] = truth_uncertainty_plot_df['log_energy_consumption']
bounds_df['upper_bound'] = test_uncertainty_plot_df['upper_bound']

bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                          (bounds_df['real_value'] <= bounds_df['upper_bound']))

print("Proportion of points contained within 99% confidence interval:",
      bounds_df['contained'].mean())