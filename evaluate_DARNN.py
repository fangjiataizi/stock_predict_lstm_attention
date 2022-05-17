from DA_RNN import da_rnn
# from dataset import getData
from parser_my import args
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from log import Logger


def eval():
    # model = torch.load(args.save_file)
    log = Logger('all.log', level='info')
    model = da_rnn(file_data='data/btc.csv', logger=log.logger, parallel=False,
                   learning_rate=.001)

    enc_checkpoint = torch.load(args.save_file_DARNN_enc)
    dec_checkpoint = torch.load(args.save_file_DARNN_dec)

    model.encoder.load_state_dict(enc_checkpoint['state_dict'])
    model.decoder.load_state_dict(dec_checkpoint['state_dict'])

    y_pred = model.predict()
    print("pred len : ", len(y_pred))
    y_true=model.y[model.train_size:]
    print("true len : ", len(y_true))






    result_t = y_true
    result_m = y_pred

    # for i in range(len(preds)):
    #     print('预测值是%.2f,真实值是%.2f' % (
    #     preds[i] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))
    #
    #     t = preds[i] * (close_max - close_min) + close_min
    #     result_t.append(t)
    #     m = labels[i] * (close_max - close_min) + close_min
    #     result_m.append(m)
    #
    # x = range(len(result_t))
    # #print(result_m)
    # #print(x)


    print(f"均方误差(MSE)：{mean_squared_error(result_t, result_m)}")
    print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(result_t, result_m))}")
    print(f"测试集R^2：{r2_score(result_t, result_m)}")

    #
    # plt.plot(x, result_t, label='pred')
    # plt.plot(x, result_m, label='true')
    #
    # plt.title("加密货币预测")  # 标题
    #
    # plt.legend()

eval()


plt.show()
