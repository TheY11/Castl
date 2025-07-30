import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sklearn
import Spanve
from Spanve import Spanve, adata_preprocess, adata_preprocess_int

warnings.filterwarnings("ignore")
sc.set_figure_params(dpi=150)
np.random.seed(2233)

datasets = ['colon2']

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    
    tissue_type = 'CRC' if dataset.startswith('colon') else 'Liver'
    
    adata_path = f"../data/CRC/ST-{dataset}/{tissue_type}_{dataset}_combined_adata.h5ad"
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()
    print(f"Initial data shape for {dataset}: {adata.shape}")

    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    print(f"After filtering for {dataset}: {adata.shape}")

    adata.layers["normalized"] = adata_preprocess(adata).X
    adata.layers['counts'] = adata.X.copy()
    adata.layers['normlized_counts'] = adata_preprocess_int(adata).X
    adata.X = adata.layers['counts']

    print(f"Running Spanve model for {dataset}...")
    svmodel = Spanve(adata, n_jobs=-1)
    svmodel.fit(verbose=True)
    print(f"Detected SV gene number for {dataset}: {svmodel.rejects.sum()}")

    result_df = svmodel.result_df
    result_df['gene'] = result_df.index
    result_df = result_df.rename(columns={'fdrs': 'adjusted_p_value'})
    result_df['pred'] = (result_df['adjusted_p_value'] < 0.05).astype(int)
    result_df['gene'] = result_df['gene'].str.replace('-', '.')

    output_path = f"../results/{tissue_type}/{dataset}/{dataset}_stabl/{tissue_type}_{dataset}_spanve_stabl_processed.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Results for {dataset} saved to: {output_path}")

print("\nAll datasets processed successfully!")