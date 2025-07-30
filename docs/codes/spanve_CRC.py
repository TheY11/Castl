import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sklearn
import Spanve
import warnings
from Spanve import Spanve, adata_preprocess, adata_preprocess_int
warnings.filterwarnings("ignore")
sc.set_figure_params(dpi=150)
np.random.seed(2233)

datasets = ['colon2']

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    tissue_type = 'CRC' if dataset.startswith('colon') else 'Liver'
    
    try:
        adata = sc.read_h5ad(f"../data/CRC/ST-{dataset}/processed_adata.h5ad")
        adata.var_names_make_unique()
        print(f"Initial shape: {adata.shape}")
        
        sc.pp.filter_cells(adata, min_counts=10)
        sc.pp.filter_genes(adata, min_cells=5)
        print(f"After filtering: {adata.shape}")
        
        adata.X = adata.X.toarray()
        adata.layers["normalized"] = adata_preprocess(adata).X
        adata.layers['counts'] = adata.X.copy()
        adata.layers['normlized_counts'] = adata_preprocess_int(adata).X
        
        adata.X = adata.layers['counts']
        svmodel = Spanve(adata, n_jobs=16)
        svmodel.fit(verbose=True)
        print(f"Detected SV gene number: {svmodel.rejects.sum()}")
        
        result_df = svmodel.result_df
        result_df['gene'] = result_df.index
        result_df = result_df.rename(columns={'fdrs': 'adjusted_p_value'})
        result_df['pred'] = (result_df['adjusted_p_value'] < 0.05).astype(int)
        result_df['gene'] = result_df['gene'].str.replace('-', '.')
        
        result_file = f"../results/{tissue_type}/{dataset}/{tissue_type}_{dataset}_spanve_results_processed.csv"
        result_df.to_csv(result_file, index=False)
        print(f"Results saved to: {result_file}")
        
    except Exception as e:
        print(f"Error processing {dataset}: {str(e)}")
        continue