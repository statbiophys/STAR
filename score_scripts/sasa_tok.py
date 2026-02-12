import torch
from tqdm import tqdm

class Vocab:
    def __init__(self, alphabet) -> None:
        self.stoi = {}
        self.itos = {}
        for i, alphabet in enumerate(alphabet):
            self.stoi[alphabet] = i
            self.itos[i] = alphabet

class TokenizerWrapper:
    def __init__(self,args, vocab, dummy_process):
        self.vocab = vocab
        self.dummy_process = dummy_process
        self.eos_token = '%'
        self.args = args
    
    def process(self, x):
        lens = [len(x[i]) for i in range(len(x))]
        if self.dummy_process:
            max_len = max(lens)
            max_len = self.args.gen_max_len
            if max_len != sum(lens) / len(lens):
                print('hello2')
                for i in range(len(x)):
                    if len(x[i]) == max_len:
                        pass
                    try:
                        x[i] = x[i] + [self.stoi[self.eos_token]] + [len(self.stoi.keys())] * (max_len - len(x[i]) - 1)
                        #x[i] = x[i] + [len(self.stoi.keys())] * (max_len - len(x[i]))
                    except:
                        import pdb; pdb.set_trace();
        else:
            ret_val = []
            max_len = max(lens)
            max_len = self.args.gen_max_len
            for i in range(len(x)):
                # process
                temp = [self.stoi[ch] for ch in x[i]]
                if max_len != sum(lens) / len(lens):
                    if len(temp) == max_len:
                        pass
                    try:
                        temp = temp + [len(self.stoi.keys())] * (max_len - len(temp))
                        #temp = temp + [len(self.stoi.keys())] * (max_len - len(temp))
                    except Exception as e:
                        print(e)
                        #import pdb; pdb.set_trace();
                ret_val.append(temp)
            x = ret_val
        return torch.tensor(x, dtype=torch.long)

    @property
    def itos(self):
        return self.vocab.itos

    @property
    def stoi(self):
        return self.vocab.stoi


def get_tokenizer(args):
    if args.task == "amp":
        alphabet = ['%', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        # %: EOS
    elif args.task == "tfbind":
        alphabet = ['A', 'C', 'T', 'G']
    elif args.task == "gfp":
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    elif args.task == "random":
        alphabet = ['%','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        #alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        #alphabet = ['A', 'C', 'D', 'E']
    
    vocab = Vocab(alphabet)
    tokenizer = TokenizerWrapper(args,vocab, dummy_process=(args.task != "random"))
    return tokenizer