import numpy as np
import matplotlib.pyplot as plt
from SASA_oracle import SASA_oracle_2
from hic_data import HICDataset
import random
from get_mery_seqs import get_mery_seqs, get_tras_seqs
import math
from iglm import IgLM

plt.rcParams.update({'font.size': 18})
name,VHseqs,VLseqs = get_mery_seqs()
_,trasVH,trasVL = get_tras_seqs()
oracle = SASA_oracle_2(use_double = True)
mery_sasa_r = oracle.compute_logits_double(VHseqs,VLseqs)
tras_sasa_r = oracle.compute_logits_double(trasVH,trasVL)

dataset = HICDataset()
mabs_sasa_r = oracle.compute_logits_double(dataset.VHseqs,dataset.VLseqs)

iglm = IgLM()

heavy_perp_mery = []
light_perp_mery = []

heavy_perp_mabs = []
light_perp_mabs = []

heavy_perp_tras = []
light_perp_tras = []

for i in range(len(VHseqs)):
    sequence = VHseqs[i]
    chain_token = "[HEAVY]"
    species_token = "[HUMAN]"
    log_likelihood = iglm.log_likelihood(
        sequence,
        chain_token,
        species_token,
    )
    perplexity = math.exp(-log_likelihood)
    heavy_perp_mery.append(perplexity)

for i in range(len(VLseqs)):
    sequence = VLseqs[i]
    chain_token = "[LIGHT]"
    species_token = "[HUMAN]"
    log_likelihood = iglm.log_likelihood(
        sequence,
        chain_token,
        species_token,
    )
    perplexity = math.exp(-log_likelihood)
    light_perp_mery.append(perplexity)

for i in range(len(dataset.VHseqs)):
    sequence = dataset.VHseqs[i]
    chain_token = "[HEAVY]"
    species_token = "[HUMAN]"
    log_likelihood = iglm.log_likelihood(
        sequence,
        chain_token,
        species_token,
    )
    perplexity = math.exp(-log_likelihood)
    heavy_perp_mabs.append(perplexity)

for i in range(len(dataset.VLseqs)):
    sequence = dataset.VLseqs[i]
    chain_token = "[LIGHT]"
    species_token = "[HUMAN]"
    log_likelihood = iglm.log_likelihood(
        sequence,
        chain_token,
        species_token,
    )
    perplexity = math.exp(-log_likelihood)
    light_perp_mabs.append(perplexity)

for i in range(len(trasVH)):
    sequence = trasVH[i]
    chain_token = "[HEAVY]"
    species_token = "[HUMAN]"
    log_likelihood = iglm.log_likelihood(
        sequence,
        chain_token,
        species_token,
    )
    perplexity = math.exp(-log_likelihood)
    heavy_perp_tras.append(perplexity)

for i in range(len(trasVL)):
    sequence = trasVL[i]
    chain_token = "[LIGHT]"
    species_token = "[HUMAN]"
    log_likelihood = iglm.log_likelihood(
        sequence,
        chain_token,
        species_token,
    )
    perplexity = math.exp(-log_likelihood)
    light_perp_tras.append(perplexity)


with open('mabs_score.csv','w') as f:
    for i in range(len(mabs_sasa_r)):
        f.write(f'{dataset.names[i]},{mabs_sasa_r[i]},{heavy_perp_mabs[i]},{light_perp_mabs[i]}\n')

with open('tras_seqs_score.csv','w') as f:
    for i in range(1):
        f.write(f'trastuzumab,{tras_sasa_r[i]},{heavy_perp_tras[i]},{light_perp_tras[i]}\n')

with open('mery_seqs_score.csv','w') as f:
    f.write('name,solubility_score,heavy_chain_perplexity,light_chain_perplexity')
    for i in range(len(mery_sasa_r)):
        f.write(f'{name[i]},{mery_sasa_r[i]},{heavy_perp_mery[i]},{light_perp_mery[i]}\n')

print('done computing the scores')