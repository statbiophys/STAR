from Levenshtein import distance
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

class Output:
    def __init__(self, df):
        self.df = df
    
    @staticmethod
    def lev_metric(x, y, data):
        i, j = int(x[0]), int(y[0]) 
        return distance(data[i], data[j])
    
    def cluster(self, threshold, cluster_min):
        top_seq = self.df[self.df['Nb_freq'] > threshold].reset_index(drop=True)
        data = top_seq['aaSeqCDR3']
        X = np.arange(len(data)).reshape(-1, 1)
        b = DBSCAN(metric=lambda x, y: self.lev_metric(x,y,data),eps=1,min_samples=cluster_min).fit(X)
        temp = pd.DataFrame(b.labels_)
        temp.reset_index(inplace=True)
        temp = temp.set_index("index").to_dict()[0]
        top_seq['family_lev'] = top_seq.index.map(temp)
        top_seq.replace(-1, np.nan, inplace=True)
        top_seq.dropna(inplace=True)
        top_seq.reset_index(drop=True, inplace=True)
        return top_seq

