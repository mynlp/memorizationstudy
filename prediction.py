import pyro
import torch
import torch.nn as nn
import torch.optim as optim
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import Normal
from pyro import distributions as dist

# 数据生成
# 创建100个样本，每个样本是一个长度20的时间序列，特征维度为10
N = 100
T = 20
D = 10
X_train = torch.randn(N, T, D).transpose(0, 1)
# y是一个N维向量，它是对每个序列输出的回归目标
Y_train = torch.randn(T, N, 1).transpose(0, 1)


# 模型定义
class BayesianLSTM(nn.Module):
    def __init__(self, n_features, n_hidden=128):
        super().__init__()

        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(n_features, n_hidden)
        self.softplus = nn.Softplus()
        self.linear_mu = nn.Linear(n_hidden, 1)
        self.linear_sigma = nn.Linear(n_hidden, 1)

    def model(self, X, y):
        X = X.transpose(0, 1)
        y = y.transpose(0, 1)
        h_0 = torch.zeros(1, X.size(1), self.n_hidden)
        c_0 = torch.zeros(1, X.size(1), self.n_hidden)

        h_n, _ = self.lstm(X, (h_0, c_0))  # 这里，我们需要的是 LSTM 的所有输出，而非最后一个
        mu = self.linear_mu(h_n).transpose(0, 1)
        sigma = self.softplus(self.linear_sigma(h_n)).transpose(0, 1)
        print("h_n shape: ", h_n.shape)
        print("mu shape: ", mu.shape)
        print("sigma shape: ", sigma.shape)
        print("obs shape: ", y.shape)
        with pyro.plate("data", X.size(1)):
            pyro.sample("obs", dist.Normal(mu, sigma.reshape(-1)).to_event(1), obs=y)

    def guide(self, X, y=None):
        X = X.transpose(0, 1)
        if y is not None:
            y = y.transpose(0, 1)
        h_0 = torch.zeros(1, X.size(1), self.n_hidden)
        c_0 = torch.zeros(1, X.size(1), self.n_hidden)

        h_n, _ = self.lstm(X, (h_0, c_0))

        mu = self.linear_mu(h_n).transpose(0, 1)
        sigma = self.softplus(self.linear_sigma(h_n)).transpose(0, 1)

        with pyro.plate("data", X.size(1)):
            pyro.sample("obs", dist.Normal(mu, sigma.reshape(-1)).to_event(1),obs=y)


# 模型训练
b_lstm = BayesianLSTM(n_features=10, n_hidden=128)

optimizer = pyro.optim.Adam({"lr": 0.01})

svi = SVI(b_lstm.model, b_lstm.guide, optimizer, loss=Trace_ELBO())

n_epochs = 100
for epoch in range(n_epochs):
    loss = svi.step(X_train, Y_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}. Loss: {loss}")


# 预测
def predict(x):
    predictive = pyro.infer.Predictive(b_lstm.model, guide=b_lstm.guide, num_samples=1)
    # 注意，我们需要给定模型的输入数据
    predicted = predictive(x)["obs"]
    # 我们只从预测的分布中采样一次，所以我们只选择第一次采样
    return predicted[0]


# 测试一下预测，并用预测值和实际值进行比较
prediction = predict(X_train)
print("Prediction: ", prediction)
print("Actual: ", Y_train)
