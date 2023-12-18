import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen
import multiprocessing
import numpy as np 
import pandas as pd

#Define the files for loading in generative model/data
class Get_pgen:
    def __init__(self):
        params_file_name = 'olga/default_models/human_B_heavy/model_params.txt'
        marginals_file_name = 'olga/default_models/human_B_heavy/model_marginals.txt'
        V_anchor_pos_file ='olga/default_models/human_B_heavy/V_gene_CDR3_anchors.csv'
        J_anchor_pos_file = 'olga/default_models/human_B_heavy/J_gene_CDR3_anchors.csv'
        genomic_data = load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        #Load model
        generative_model = load_model.GenerativeModelVDJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        #Process model/data for pgen computation by instantiating GenerationProbabilityVDJ
        self.pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)

    def compute_pgen(self,a):
        return self.pgen_model.compute_aa_CDR3_pgen(a)
    
    def compute_pgen_parallel(self,seq):
        return seq.apply(self.compute_pgen)
    
    def apply_parallel(self,df):
        num_processes = 10
        pool = multiprocessing.Pool(processes=num_processes)
        chunks = np.array_split(df, num_processes)
        results = pool.map(self.compute_pgen_parallel, chunks)
        merged_results = pd.concat(results)

        return merged_results

