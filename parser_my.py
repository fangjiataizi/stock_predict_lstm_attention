import argparse  #argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
import torch

parser = argparse.ArgumentParser()   #创建一个解析对象

# parser.add_argument('--corpusFile', default='data/000001SH_index.csv')  #向该对象中添加你要关注的命令行参数和选项
parser.add_argument('--corpusFile', default='data/btc.csv')

# TODO 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=1000, type=int) # 训练轮数
# parser.add_argument('--epochs', default=10, type=int) # 训练轮数
parser.add_argument('--layers', default=2, type=int) # LSTM层数
parser.add_argument('--input_size', default=5, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=32, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
parser.add_argument('--sequence_length', default=5, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--useGPU', default=False, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--save_file', default='model/stock.pkl') # 模型保存位置
parser.add_argument('--save_file_RNN', default='model/RNN_stock.pkl') # 模型保存位置


# args = parser.parse_args()
args, unknown = parser.parse_known_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device