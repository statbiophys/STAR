import os
import numpy as np
import pandas as pd
from numba import jit
import atriegc


class Get_df:
    def __init__(self, data):
        self.data = data
    
    def frequency(self):
        df_freq=pd.DataFrame(self.data["aaSeqCDR3"])
        df_freq["size"]=1
        n_df=len(df_freq)
        temp=df_freq.groupby(['aaSeqCDR3'], sort = False).sum()
        temp["frequency"]=temp["size"]/n_df
        temp=temp.reset_index()
        temp=temp.set_index("aaSeqCDR3").to_dict()["frequency"]
        return temp
        
        
    def multiplicity(self):
        unique_nucl = self.data.drop_duplicates("nSeqCDR3")
        unique_nucl.reset_index(drop = True, inplace = True)
        unique_nucl=unique_nucl.assign(size=1)
        n_df=len(unique_nucl)
        temp=unique_nucl.groupby(['aaSeqCDR3'], sort = False).sum()
        temp=temp.reset_index()
        temp=temp.set_index("aaSeqCDR3").to_dict()["size"]
        return temp, len(unique_nucl)
        
        
    def neighbours(self):
        distance =1
        tr = atriegc.TrieAA()
        df_nb=pd.DataFrame(self.data["aaSeqCDR3"])
        df_nb.drop_duplicates("aaSeqCDR3",inplace=True)
        df_nb.reset_index(drop=True,inplace=True)
        df_nb=df_nb.assign(nb_neighbours_real=0)
        df_nb=df_nb.assign(nb_freq=0)
        n_df=len(df_nb)
        for k1 in range(n_df):
            tr.insert(df_nb["aaSeqCDR3"][k1])
        for k1 in range(n_df):    
            a=(tr.neighbours(df_nb["aaSeqCDR3"][k1], distance))
            c=0
            for k in range(len(a)):
                d=a[k]
                dic,n_un=self.multiplicity()
                b=int(dic[d])
                c+=b
            df_nb.loc[k1,"nb_neighbours_real"]=c-1
            df_nb.loc[k1,"nb_freq"]=(c-1)/n_un
        temp=df_nb.set_index("aaSeqCDR3").to_dict()["nb_neighbours_real"]
        temp1=df_nb.set_index("aaSeqCDR3").to_dict()["nb_freq"]
        return temp,temp1
    
    def make(self):
        df_read=pd.DataFrame(self.data["aaSeqCDR3"])
        df_read.drop_duplicates("aaSeqCDR3",inplace=True)
        df_read.reset_index(drop=True,inplace=True)
        dic_1 = self.frequency()
        dic_2 = self.multiplicity()[0]
        dic_3, dic_4 = self.neighbours()
        df_read["Frequency"]=df_read.aaSeqCDR3.map(dic_1)
        df_read["Multiplicity"]=df_read.aaSeqCDR3.map(dic_2)
        df_read["Neighbours"]=df_read.aaSeqCDR3.map(dic_3)
        df_read["Nb_freq"]=df_read.aaSeqCDR3.map(dic_4)
        return df_read
    