import pandas as pd
from numba import jit
import math as mt
import numpy as np

@jit(nopython=True)
def pvalue(n_gen,n_tag,df_gen,df_real,df_pvalue):
    q=2
    for k1 in range(n_tag):
        a = (n_tag*q*(df_gen[k1] + 0.1))/(n_gen)
        for d in range(df_real[k1],1000,1):
            b = -a + d*mt.log(a) - mt.lgamma(d+1)
            df_pvalue[k1]+= np.exp(b)

class Output_MC:
    def __init__(self, df):
        self.df = df
        
    def get_len(self):
        self.df["CDR3_len"]=self.df["aaSeqCDR3"].str.len()
        
    def get_lambda(self,uniq_nucl):
        self.get_len()
        self.df["Lambda_freq"]=self.df["CDR3_len"]*19*self.df["Pgen"]
        self.df["Lambda"]=self.df["Lambda_freq"]*uniq_nucl
                    
    def get_pvalue(self,uniq_nucl):
        self.get_lambda(uniq_nucl)
        n_gen=len(self.df)
        nb_gen="Lambda"
        nb_real="Neighbours"
        n_tag=len(self.df)
        df_nb_gen=np.zeros(n_tag)
        df_nb_real=np.zeros(n_tag)
        df_pvalue=np.zeros(n_tag)
        df_nb_gen=self.df[nb_gen].values
        df_nb_real=self.df[nb_real].values
        pvalue(n_gen,n_tag,df_nb_gen,df_nb_real,df_pvalue)    
        self.df["Pvalue"]=df_pvalue
    
    def BH_procedure(self,uniq_nucl,alpha):
        self.get_pvalue(uniq_nucl)
        self.df.sort_values(by="Pvalue",inplace=True)
        self.df.reset_index(drop=True,inplace=True)
        m=len(self.df)
        for k in range(1,m):
            if self.df["Pvalue"][k]>(k/m)*alpha:
                top_seq=self.df[0:k]
                break
        return top_seq
    
    

   