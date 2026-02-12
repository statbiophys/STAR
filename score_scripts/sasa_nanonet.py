import numpy as np
from torch import nn
from torch import from_numpy
from torch.cuda import is_available
import torch
import math

class args:
    def __init__(self):
        self.task = 'random'
        self.gen_max_len = 180
        self.num_tokens = 22
        self.small_embedding = 8
        self.pad_token_id = 21
        if is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.hidden_size = 64
        self.hidden_size_2 = 64
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.n_layers = 1
        self.max_r = 16
        self.epochs = 20
        self.model = 'cnn'

def _same_pad(k=1, dil=1):
    # assumes stride length of 1
    # p = math.ceil((l - 1) * s - l + dil*(k - 1) + 1)
    p = math.ceil(dil*(k - 1))
    #print("padding:", p)
    return p


class AbEmbeddings(torch.nn.Module):
    """
    Residue embedding and Positional embedding
    """
    
    def __init__(self, args):
        super().__init__()
        self.pad_token_id = args.pad_token_id
        
        self.AAEmbeddings = torch.nn.Embedding(args.num_tokens, args.small_embedding, padding_idx=self.pad_token_id)
        self.Dropout = torch.nn.Dropout(args.hidden_dropout_prob)
        self.UpEmbedding = torch.nn.Linear(args.small_embedding,args.hidden_size)
        self.args = args

    def forward(self, src):
        inputs_embeds = self.AAEmbeddings(src)
        return self.Dropout(self.UpEmbedding(inputs_embeds))

class ByteNetBlock(nn.Module):
    def __init__(self,args,k,r,types):
        super(ByteNetBlock,self).__init__()
        if types == 1:
            self.conv_1 = torch.nn.Conv1d(in_channels = args.hidden_size, out_channels = args.hidden_size, kernel_size = k, dilation = r)
            self.conv_2 = torch.nn.Conv1d(in_channels = args.hidden_size, out_channels = args.hidden_size, kernel_size = k, dilation = r)
            self.conv_3 = torch.nn.Conv1d(in_channels = args.hidden_size, out_channels = args.hidden_size, kernel_size = k, dilation = r)
            p = _same_pad(k,r)
            if p % 2 == 1:
                padding = [p // 2 + 1, p // 2]
            else:
                padding = (p // 2, p // 2)
            self.pad = nn.ConstantPad1d(padding, 0.)
            self.batch_norm = torch.nn.BatchNorm1d(args.hidden_size,eps = args.layer_norm_eps)
            self.RELU = torch.nn.ReLU()
            self.RELU_2 = torch.nn.ReLU()
        else:
            self.conv_1 = torch.nn.Conv1d(in_channels = args.hidden_size*2, out_channels = args.hidden_size*2, kernel_size = k, dilation = r)
            self.conv_2 = torch.nn.Conv1d(in_channels = args.hidden_size*2, out_channels = args.hidden_size*2, kernel_size = k, dilation = r)
            self.conv_3 = torch.nn.Conv1d(in_channels = args.hidden_size*2, out_channels = args.hidden_size*2, kernel_size = k, dilation = r)
            p = _same_pad(k,r)
            if p % 2 == 1:
                padding = [p // 2 + 1, p // 2]
            else:
                padding = (p // 2, p // 2)
            self.pad = nn.ConstantPad1d(padding, 0.)
            self.batch_norm = torch.nn.BatchNorm1d(args.hidden_size*2,eps = args.layer_norm_eps)
            self.RELU = torch.nn.ReLU()
            self.RELU_2 = torch.nn.ReLU()

    def forward(self,x):
        x = self.pad(x)
        x = self.conv_1(x)
        y = x
        x = self.pad(x)
        x = self.conv_2(x)
        x = self.RELU(x)
        x = self.pad(x)
        x = self.conv_3(x)
        x = self.batch_norm(x)
        x = x + y
        x = self.RELU_2(x)
        return x

class ByteNetDecoder2(torch.nn.Module):
    """
    Head for masked sequence prediction.
    """

    def __init__(self, args):
        super(ByteNetDecoder2,self).__init__()
        self.dense = torch.nn.Linear(args.hidden_size*2, args.hidden_size)
        self.RELU = torch.nn.ReLU()
        self.decoder = torch.nn.Linear(args.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.RELU(x)
        x = self.decoder(x)
        return x

class NanoNet(nn.Module):
    def __init__(self,args):
        super(NanoNet,self).__init__()
        self.EmbeddingLayer = AbEmbeddings(args)
        self.k_1 = 25
        self.r_1 = 1
        self.num_b_1 = 3
        self.k_2 = 5
        self.num_b_2 = 5
        self.r_list = [2**i if 2**i <= args.max_r else args.max_r for i in range(self.num_b_2)]
        self.blocks_1 = nn.Sequential(*[ByteNetBlock(args,self.k_1,self.r_1,1) for _ in range(self.num_b_1)])
        self.blocks_2 = nn.Sequential(*[ByteNetBlock(args,self.k_2,i,2) for i in self.r_list])
        self.decoder_2 = ByteNetDecoder2(args)
        self.conv = torch.nn.Conv1d(in_channels = args.hidden_size, out_channels = args.hidden_size*2, kernel_size = 1)
        self.Dropout = torch.nn.Dropout(0.25)

    def forward(self,x):
        x = self.EmbeddingLayer(x)
        x = torch.transpose(x,1,2)
        x = self.blocks_1(x)
        x = self.conv(x)
        x = self.blocks_2(x)
        x = torch.transpose(x,1,2)
        x = self.Dropout(x)
        x = self.decoder_2(x)
        return x

    def freeze_blocks(self):
        for params in self.blocks.parameters():
            params.requires_grad = False

    def unfreeze_blocks(self):
        for params in self.blocks.parameters():
            params.requires_grad = True