import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from numpy import random

class HICDataset(Dataset):

    def __init__(self):
        self.good_names = ['abrilumab', 'alirocumab', 'bapineuzumab', 'bavituximab', 'benralizumab', 'blosozumab', 'brentuximab', 'brodalumab', 'canakinumab', 'carlumab', 'cixutumumab', 'clazakizumab', 'codrituzumab', 'dacetuzumab', 'dalotuzumab', 'denosumab', 'dinutuximab', 'duligotuzumab', 'eculizumab', 'eldelumab', 'elotuzumab', 'emibetuzumab', 'enokizumab', 'epratuzumab', 'etrolizumab', 'evolocumab', 'farletuzumab', 'fasinumab', 'fezakinumab', 'ficlatuzumab', 'figitumumab', 'fletikumab', 'foralumab', 'fresolimumab', 'fulranumab', 'galiximab', 'ganitumab', 'gemtuzumab', 'girentuximab', 'glembatumumab', 'imgatuzumab', 'inotuzumab', 'lampalizumab', 'lenzilumab', 'lintuzumab', 'lirilumab', 'lumiliximab', 'mavrilimumab', 'mepolizumab', 'mogamulizumab', 'natalizumab', 'nimotuzumab', 'ocrelizumab', 'ofatumumab', 'olaratumab', 'otelixizumab', 'otlertuzumab', 'ozanezumab', 'palivizumab', 'panobacumab', 'parsatuzumab', 'patritumab', 'pinatuzumab', 'polatuzumab', 'radretumab', 'reslizumab', 'rilotumumab', 'robatumumab', 'romosozumab', 'sarilumab', 'seribantumab', 'siltuximab', 'simtuzumab', 'sirukumab', 'tabalumab', 'teplizumab', 'tigatuzumab', 'tildrakizumab', 'tocilizumab', 'tovetumab', 'vedolizumab', 'veltuzumab', 'visilizumab', 'zalutumumab', 'zanolimumab']
        self.data = pd.read_csv('./hic.csv').values
        self.names = self.data[:,0]
        self.seqs = self.data[:,1]
        self.VHseqs = self.data[:,1]
        self.VLseqs = self.data[:,2]
        self.hic = self.data[:,6]
        idx = [i for i in range(len(self.seqs)) if self.hic[i] < 15]
        self.seqs = [self.seqs[i] for i in idx if self.names[i] in self.good_names]
        self.VHseqs = [self.VHseqs[i] for i in idx if self.names[i] in self.good_names]
        self.VLseqs = [self.VLseqs[i] for i in idx if self.names[i] in self.good_names]
        self.hic = [self.hic[i] for i in idx if self.names[i] in self.good_names]
        self.names = [self.names[i] for i in idx if self.names[i] in self.good_names]
        self.hic = np.array(self.hic)
        self.extract_camsol()
        self.camsol_score = [self.camsol_score[i] for i in range(len(self.camsol_names)) if self.camsol_names[i] in self.good_names]
        self.camsol_per_aa_score = [self.camsol_per_aa_score[i] for i in range(len(self.camsol_names)) if self.camsol_names[i] in self.good_names]

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):

        return self.seqs[idx], self.hic[idx]

    def write_seqs(self):
        with open('./seqs.fa','w') as f:
            for i in range(len(self.seqs)):
                f.write('> {}\n'.format(self.names[i]))
                f.write('{}\n'.format(self.seqs[i]))

    def extract_camsol(self):
        self.camsol_names = []
        self.camsol_score = []
        self.camsol_per_aa_score = []
        with open('./camsol.tsv','r') as f:
            next(f)
            for line in f:
                a = line.split('\t')
                self.camsol_names.append(a[0])
                self.camsol_score.append(float(a[1]))
                self.camsol_per_aa_score.append(a[2].split(';'))
                self.camsol_per_aa_score[-1][-1] = self.camsol_per_aa_score[-1][-1].split('\n')[0]
                self.camsol_per_aa_score[-1] = np.array([float(i) for i in self.camsol_per_aa_score[-1]])
        self.camsol_score = -np.array(self.camsol_score)

    def read_fasta(self,file_path):
        fasta_dict = {}
        with open(file_path, 'r') as file:
            sequence_id = ""
            sequence = ""
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if sequence_id:
                        fasta_dict[sequence_id] = sequence
                    sequence_id = line[1:]  # Exclude the ">" character
                    sequence = ""
                else:
                    sequence += line
            if sequence_id:
                fasta_dict[sequence_id] = sequence
        return fasta_dict

    def write_fasta(self,fasta_dict, output_path):
        with open(output_path, 'w') as file:
            for sequence_id, sequence in fasta_dict.items():
                if sequence_id[1:] in self.good_names:
                    file.write(f">{sequence_id[1:]}\n")
                    file.write(f"{sequence}\n")
    
    def write_fasta_2(self,output_path):
        with open(output_path, 'w') as file:
            for i in range(len(self.names)):
                if self.names[i] in self.good_names:
                    file.write(f">{self.names[i]}\n")
                    file.write(f"{self.seqs[i]}\n")

if __name__ == '__main__':
    d = HICDataset()
    print(len(d.camsol_score))