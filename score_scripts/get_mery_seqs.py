from pandas import read_csv

def get_mery_seqs():
    data = read_csv('./mery_seqs.tsv', delimiter = '\t')[['Name','aaSeqHeavy','aaSeqLight']].values
    name = [d[0] for d in data]
    heavy_seqs = [d[1] for d in data]
    light_seqs = [d[2] for d in data]
    return name,heavy_seqs,light_seqs

def get_tras_seqs():
    name = 'trastuzumab'
    heavy_chain = 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS'
    light_chain =  'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK'
    name = [name for i in range(10)]
    heavy_seqs = [heavy_chain for i in range(10)]
    light_seqs = [light_chain for i in range(10)]
    return name,heavy_seqs,light_seqs
    