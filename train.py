from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel_org import lstm
from parser_my import args
from dataset import getData
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train():

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first) #创建LSTM()类的一个对象
    model.to(args.device) #调用可用设备
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001，对于优化器函数，我们将使用adam优化器

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size )
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                pred = pred[1, :, :]  #取第一维
                label = label.unsqueeze(1)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss)
        if i % 10 == 0:
            # torch.save(model, args.save_file)
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('第%d epoch，保存模型' % i)
    # torch.save(model, args.save_file)
    torch.save({'state_dict': model.state_dict()}, args.save_file)

train()