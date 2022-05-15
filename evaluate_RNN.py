from RNN_base import RNN_BASE
from dataset import getData
from parser_my import args
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def eval():
    # model = torch.load(args.save_file)
    model = RNN_BASE(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file_RNN)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    result_t = []
    result_m = []

    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' % (
        preds[i] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))

        t = preds[i] * (close_max - close_min) + close_min
        result_t.append(t)
        m = labels[i] * (close_max - close_min) + close_min
        result_m.append(m)

    x = range(len(result_t))
    #print(result_m)
    #print(x)


    print(f"均方误差(MSE)：{mean_squared_error(result_t, result_m)}")
    print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(result_t, result_m))}")
    print(f"测试集R^2：{r2_score(result_t, result_m)}")


    plt.plot(x, result_t, label='pred')
    plt.plot(x, result_m, label='true')

    plt.title("加密货币预测")  # 标题

    plt.legend()

eval()


plt.show()
