import torch.nn as nn   #torch.nn是pytorch中自带的一个函数库，里面包含了神经网络中使用的一些常用函数
import torch.nn.init as init



def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

class RNN_BASE(nn.Module):

    def __init__(self, input_size=8, hidden_size=32, num_layers=1, output_size=1, dropout=0, batch_first=True,weight_init=None):
        super(RNN_BASE, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        # 只需要使用apply函数即可
        # torch.manual_seed(2021) # 设置随机种子
        self.apply(init_weights)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        #self.bn = nn.BatchNorm1d(32)
        #self.activation = nn.LeakyReLU(0,1)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))
        out = self.linear(hidden)
        #out = self.bn(out)
        #out = self.activation(out)

        return out