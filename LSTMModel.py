import torch.nn as nn   #torch.nn是pytorch中自带的一个函数库，里面包含了神经网络中使用的一些常用函数
import torch.nn.init as init
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

class lstm(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_layers=1, output_size=1, dropout=0, batch_first=True,weight_init=None):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        # 只需要使用apply函数即可
        # torch.manual_seed(2021) # 设置随机种子
        self.window_size=5
        self.batch_size=1

        self.apply(init_weights)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        #self.bn = nn.BatchNorm1d(32)
        #self.activation = nn.LeakyReLU(0,1)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, next_hidden_state):

        attn_weights = torch.bmm(lstm_output.reshape(self.batch_size, self.window_size, self.hidden_size), next_hidden_state.reshape(self.batch_size,self.hidden_size,self.num_layers))
        # print('attn_weights,',attn_weights.shape)
        # print(F.softmax(attn_weights, dim = 1).shape)
        soft_attn_weights = F.softmax(attn_weights, dim = 1).reshape(self.batch_size, self.num_layers, self.window_size)
        # print('soft_attn_weights,',soft_attn_weights.shape)
        # print(torch.transpose(lstm_output,0,1).shape)
        # new_hidden_state = torch.bmm(soft_attn_weights, torch.transpose(lstm_output,0,1))
        new_hidden_state = torch.bmm(soft_attn_weights, lstm_output)
        #https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
        # print('atten net out shape:',new_hidden_state.shape)
        return torch.transpose(new_hidden_state,0,1)

    def forward(self, x):
        # print (x.shape)
        # x = x.permute(1, 0, 2)
        # print (x.shape)
        # output, (final_hidden_state, final_cell_state) = self.rnn(x)  # x.shape : batch, seq_len, input_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        all_hidden_states, (next_hidden_state, next_cell_state) = self.rnn(x)
        # print ('next hidden size',next_hidden_state.shape)
        # print ('all hidden size',all_hidden_states.shape)
        next_hidden_state_att = self.attention_net(all_hidden_states, next_hidden_state)
        # print ('next hidden size',next_hidden_state_att.shape)
        # out = self.linear(next_hidden_state)
        out = self.linear(next_hidden_state_att)

        #out = self.bn(out)
        #out = self.activation(out)
        # print('out size :',out.shape)
        return out




# # define our own model which is an lstm followed by two dense layers
# class lstm(nn.Module):
#     def __init__(self, input_size=8, hidden_size=32, num_layers=1, output_size=1, dropout=0, batch_first=True,weight_init=None):
#         super(lstm, self).__init__()
#         # lstm的输入 #batch,seq_len, input_size
#         # 只需要使用apply函数即可
#         # torch.manual_seed(2021) # 设置随机种子
#         self.apply(init_weights)
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.num_layers = num_layers
#         self.output_size = output_size
#         self.dropout = dropout
#         self.batch_first = batch_first

#         # self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
#         # self.embedding.weight.requires_grad = False
#         self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )

#         # self.lstm1 = nn.LSTM(input_size=self.embedding.embedding_dim,
#         #                     hidden_size=hidden_dim,
#         #                     num_layers=1, 
#         #                     bidirectional=True)
#         self.atten1 = Attention(self.input_size, batch_first=True) # 2 is bidrectional
#         self.lstm2 = nn.LSTM(input_size=hidden_size*2,
#                             hidden_size=hidden_size,
#                             num_layers=1, 
#                             bidirectional=True)
#         self.atten2 = Attention(hidden_size*2, batch_first=True)
#         self.fc1 = nn.Sequential(nn.Linear(hidden_size*num_layers*2, hidden_size*num_layers*2),
#                                  nn.BatchNorm1d(hidden_size*num_layers*2),
#                                  nn.ReLU()) 
#         self.fc2 = nn.Linear(hidden_size*num_layers*2, 1)

#     def forward(self, x):
#         # x = self.embedding(x)
#         # x = self.dropout(x)
#         print ("-----sizze:",x.shape)
#         # x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
#         out1, (h_n, c_n) = self.lstm1(x)
#         # x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
#         x, _ = self.atten1(x) # skip connect

#         out2, (h_n, c_n) = self.lstm2(out1)
#         # y, lengths = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
#         y, _ = self.atten2(y)
        
#         z = torch.cat([x, y], dim=1)
#         z = self.fc1(self.dropout(z))
#         z = self.fc2(self.dropout(z))
#         return z




# ##attention layer 
# class Attention(nn.Module):
#     def __init__(self, hidden_size, batch_first=False):
#         super(Attention, self).__init__()

#         self.hidden_size = hidden_size
#         self.batch_first = batch_first

#         self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

#         stdv = 1.0 / np.sqrt(self.hidden_size)
#         for weight in self.att_weights:
#             nn.init.uniform_(weight, -stdv, stdv)

#     def get_mask(self):
#         pass

#     # def forward(self, inputs, lengths):
#     def forward(self, inputs):
#         if self.batch_first:
#             batch_size, max_len = inputs.size()[:2]
#         else:
#             max_len, batch_size = inputs.size()[:2]
            
        
#         # apply attention layer
#         weights = torch.bmm(inputs,
#                             self.att_weights  # (1, hidden_size)
#                             .permute(1, 0)  # (hidden_size, 1)
#                             .unsqueeze(0)  # (1, hidden_size, 1)
#                             .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
#                             )
    
#         attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

#         # # create mask based on the sentence lengths
#         # mask = torch.ones(attentions.size(), requires_grad=True).cuda()
#         # for i, l in enumerate(lengths):  # skip the first sentence
#         #     if l < max_len:
#         #         mask[i, l:] = 0

#         # # apply mask and renormalize attention scores (weights)
#         # masked = attentions * mask
#         # _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
#         # attentions = masked.div(_sums)

#         # apply attention weights
#         weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

#         # get the final fixed vector representations of the sentences
#         representations = weighted.sum(1).squeeze()

#         return representations, attentions