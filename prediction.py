import numpy as np
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# 参数设置
time_steps = 120  # 总的时间步
seq_length = 10   # 每个训练样本的序列长度

# 生成合成数据
data = torch.sin(torch.arange(0, time_steps, 0.1))  # 使用sin函数生成数据作为示例
data = data.view(-1, 1)  # 调整形状，即(time_steps, 1)

# 创建训练数据和测试数据
X = torch.zeros((time_steps - seq_length, seq_length, 1))
y = torch.zeros((time_steps - seq_length, 1))

for i in range(0, time_steps - seq_length):
    X[i] = data[i:i+seq_length]
    y[i] = data[i+seq_length]

split = int((time_steps - seq_length) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import torch.nn as nn


class BayesianLSTM(PyroModule):
    def __init__(self, input_size=1, hidden_size=50, out_size=1):
        super().__init__()
        self.lstm = PyroModule[nn.LSTM](input_size, hidden_size, batch_first=True).cuda()
        self.lstm.weight_ih_l0 = PyroSample(dist.Normal(0., 1).expand([4 * hidden_size, input_size]).to_event(2))
        self.lstm.weight_hh_l0 = PyroSample(dist.Normal(0., 1).expand([4 * hidden_size, hidden_size]).to_event(2))
        self.linear = PyroModule[nn.Linear](hidden_size, out_size)
        self.linear.weight = PyroSample(dist.Normal(0., 1).expand([out_size, hidden_size]).to_event(2))

    def forward(self, x, y=None):
        x, _ = self.lstm(x.cuda())
        x = self.linear(x[:, -1, :])
        return x


from pyro.infer import SVI, Trace_ELBO
from pyro.optim import AdagradRMSProp
from pyro.infer.autoguide import AutoDiagonalNormal



def train(model, X_train, y_train, num_steps=300000):
    optim = AdagradRMSProp({"eta": 1e-2})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    for step in range(num_steps):
        loss = svi.step(X_train.cuda(), y_train.cuda())
        if step % 1000 == 0:
            print(f"Step {step} : loss = {loss}")


model = BayesianLSTM()
model = model.cuda()
guide = AutoDiagonalNormal(model)
train(model, X_train.cuda(), y_train.cuda())

from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

# 生成后验样本
predictive = Predictive(model, guide=guide, num_samples=800,
                        return_sites=("_RETURN",))
samples = predictive(X_test)
predictions = samples["_RETURN"]

# 计算预测的均值和标准差作为不确定性度量
mean_predictions = predictions.mean(0)
std_predictions = predictions.std(0)

# 可视化
import matplotlib.pyplot as plt

plt.fill_between(np.arange(len(y_test)),
                 mean_predictions.squeeze() - std_predictions.squeeze(),
                 mean_predictions.squeeze() + std_predictions.squeeze(), alpha=0.3, color='r', label="Uncertainty Interval")
plt.plot(np.arange(len(y_test)), mean_predictions.squeeze(), label="Predicted Mean")
plt.plot(np.arange(len(y_test)), y_test.squeeze(), label="True", alpha=0.5)
plt.legend()
plt.show()