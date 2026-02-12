import numpy as np
from torch import from_numpy
from torch.cuda import is_available
import torch

import sys
from small_model import small_nn
from sasa_nanonet import NanoNet
from sasa_tok import get_tokenizer
import os


class args:
    def __init__(self):
        self.task = 'random'
        self.gen_max_len = 180
        self.num_tokens = 22
        self.small_embedding = 16
        self.pad_token_id = 21
        if is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.hidden_size = 64
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.max_r = 16
        self.epochs = 20
        self.model = 'cnn'

class SASA_oracle_2:
    def __init__(self,use_double = False):
        self.weights = {'alpha': -9.44 ,'F': 2.764,'I':1.724, 'L': 1.72, 'Y': 2.059, 'W': 3.132, 'V': 1.085, 'M':0.551, 'P': 0.486,
        'C':0.61, 'A':0.064, 'G': 0, 'T':0.181, 'S':0.076, 'K':-1.198, 'Q': 0.516, 'N':0.143, 'E': -0.807, 'D': -0.476,
        'R':-0.358,'H':0.028}
        if is_available():  
            self.dev = "cuda:0" 
        else:  
            self.dev = "cpu"  
        self.args = args()
        self.small_model = NanoNet(self.args).to(self.dev)
        script_dir = os.path.dirname(__file__)
        if use_double:
            file_path = os.path.join(script_dir,'model_parameters', 'best_double.pt')
        else:
            file_path = os.path.join(script_dir,'model_parameters', 'best.pt')
        self.small_model.load_state_dict(torch.load(file_path,map_location=torch.device(self.dev)))
        self.small_model.eval()
        self.tok  = get_tokenizer(self.args)

    def compute_SASA(self,VHseqs):
        seqs = self.tok.process(VHseqs).to(self.dev)
        with torch.no_grad():
            SASA = self.small_model(seqs).squeeze()
        return SASA

    def compute_SASA_double(self,VHseqs,VLseqs):
        VHseqs = self.tok.process(VHseqs).to(self.dev)
        VLseqs = self.tok.process(VLseqs).to(self.dev)
        seqs = torch.cat((VHseqs,VLseqs),dim = 1)
        with torch.no_grad():
            SASA = self.small_model(seqs).squeeze()
        return SASA

    def compute_logits(self,VHseqs):
        p = []
        lengths = [len(i) for i in VHseqs]
        for i in VHseqs:
            w = [self.weights[j] for j in i]
            p.append(np.array(w))
        SASA = self.compute_SASA(VHseqs).cpu()
        score = []
        for i in range(len(p)):
            score.append((p[i].dot(SASA[i][:p[i].size]) + self.weights['alpha']))
        score = np.array(score)
        return score

    def compute_logits_double(self,VHseqs,VLseqs):
        p_VH = []
        p_VL = []
        lengths_VH = [len(i) for i in VHseqs]
        lengths_VL = [len(i) for i in VLseqs]
        for i in VHseqs:
            w_VH = [self.weights[j] for j in i]
            p_VH.append(np.array(w_VH))
        for i in VLseqs:
            w_VL = [self.weights[j] for j in i]
            p_VL.append(np.array(w_VL))    
        SASA = self.compute_SASA_double(VHseqs,VLseqs).cpu()
        VHSASA = SASA[:,:180]
        VLSASA = SASA[:,180:]
        score = []
        for i in range(len(p_VH)):
            score.append((p_VH[i].dot(VHSASA[i][:p_VH[i].size])+ p_VL[i].dot(VLSASA[i][:p_VL[i].size]) + self.weights['alpha']))
        score = np.array(score)
        return score

    def compute_score_without_SASA(self,VHseqs):
        p = []
        for i in VHseqs:
            w = [self.weights[j] for j in i]
            p.append(np.sum(w))
        return p

    def use_custom_weights(self,VHseqs,c_w):
        SASA = self.compute_SASA(VHseqs).cpu()
        score = []
        for i in range(len(c_w)):
            score.append((c_w[i].dot(SASA[i][:c_w[i].size])))
        score = np.array(score)
        return score

    def compute_score_double(self,VHseqs, VLseqs):
        for i in range(0,len(VHseqs)):
            if len(VHseqs[i]) == 0:
                VHseqs[i] = '%'
            if VHseqs[i][-1] == '%':
                VHseqs[i] = VHseqs[i][:-1]
        for i in range(0,len(VLseqs)):
            if len(VLseqs[i]) == 0:
                VLseqs[i] = '%'
            if VLseqs[i][-1] == '%':
                VLseqs[i] = VLseqs[i][:-1]
        c = self.compute_logits_double(VHseqs)
        return c

    def __call__(self,VHseqs):
        for i in range(0,len(VHseqs)):
            if len(VHseqs[i]) == 0:
                VHseqs[i] = '%'
            if VHseqs[i][-1] == '%':
                VHseqs[i] = VHseqs[i][:-1]
        c = self.compute_logits(VHseqs)
        return c


if __name__ == '__main__':
    #compute_thera_rescod()
    #compute_SASA_from_rescod()
    #compute_logits()
    torch.manual_seed(0)
    VHseqs = ['QVQLVESGGGVVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVSVIYSGGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCPPYQVPGPDAFDIWGQGTMVTVSS',
        'QVQLVESGGGVVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVSVIYSGGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYY'+'CAKCFFFFFFFFDIW'+'GQGTMVTVSS',
        'QVQLVESGGGVVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVSVIYSGGSSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCPPYQVPGPDAFDIWGQGTMVTVSS']

    a = SASA_oracle_2()
    print(a(VHseqs))