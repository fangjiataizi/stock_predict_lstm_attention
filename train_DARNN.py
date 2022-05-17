from torch.autograd import Variable
import torch.nn as nn
import torch
from DA_RNN import da_rnn
from parser_my import args

from log import Logger

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train():
    log = Logger('all.log', level='info')
    model = da_rnn(file_data='data/btc.csv', logger=log.logger, parallel=False,
                   learning_rate=.001)

    model.train(n_epochs=20)
    print('start save model')
    torch.save({'state_dict': model.encoder.state_dict()}, args.save_file_DARNN_enc)
    torch.save({'state_dict': model.decoder.state_dict()}, args.save_file_DARNN_dec)
    print('end save model')

train()