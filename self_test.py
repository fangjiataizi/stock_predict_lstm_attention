from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size )


