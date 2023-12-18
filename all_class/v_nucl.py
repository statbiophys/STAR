import pandas as pd

class V_nucl:
    def __init__(self, df):
        self.df = df
    
    def get_vj(self):
        df_vj = self.df.drop_duplicates(['aaSeqCDR3', 'VGene']).copy()
        df_vj.loc[:, "V_usage"] = df_vj.duplicated("aaSeqCDR3")
        df_vj = df_vj.groupby(['aaSeqCDR3'], sort=False).sum()
        df_vj = df_vj.reset_index()
        df_vj.loc[:, "V_usage"] = df_vj["V_usage"] + 1
        temp = df_vj.set_index("aaSeqCDR3").to_dict()["V_usage"]
        return temp
        
    def get_nucl(self):
        df_nucl = self.df.drop_duplicates(['nSeqCDR3', 'aaSeqCDR3']).copy()
        df_nucl.loc[:, "Nucl_usage"] = df_nucl.duplicated("aaSeqCDR3")
        df_nucl = df_nucl.groupby(['aaSeqCDR3'], sort=False).sum()
        df_nucl = df_nucl.reset_index()
        df_nucl.loc[:, "Nucl_usage"] = df_nucl["Nucl_usage"] + 1
        temp = df_nucl.set_index("aaSeqCDR3").to_dict()["Nucl_usage"]
        return temp
    
    def make_vnucl(self):
        df_read = pd.DataFrame(self.df["aaSeqCDR3"])
        df_read.drop_duplicates("aaSeqCDR3", inplace=True)
        df_read.reset_index(drop=True, inplace=True)
        dic_1 = self.get_vj()
        dic_2 = self.get_nucl()
        df_read["V_mult"] = df_read.aaSeqCDR3.map(dic_1)
        df_read["Nucl_mult"] = df_read.aaSeqCDR3.map(dic_2)
        return df_read