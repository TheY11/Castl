import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
from somde import SomNode
from somde.util import *

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

datasets = ['colon2']

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    tissue_type = 'CRC' if dataset.startswith('colon') else 'Liver'
    
    try:
        adata = sc.read_h5ad(f"../data/CRC/ST-{dataset}/processed_adata.h5ad")
        adata.var_names_make_unique()
        print(f"Initial data dimensions: {adata.shape}")
        
        sc.pp.filter_cells(adata, min_counts=10)
        sc.pp.filter_genes(adata, min_cells=5)
        print(f"Dimensions after filtering: {adata.shape}")
        
        express_matrix = pd.DataFrame(adata.X.toarray(), 
                                    index=adata.obs_names, 
                                    columns=adata.var_names)
        coordinates = adata.obsm['spatial']
        coordinates = pd.DataFrame(coordinates, 
                                 index=adata.obs_names, 
                                 columns=['x', 'y'])
        
        coordinates['total_counts'] = express_matrix.sum(1)
        express_matrix = express_matrix.loc[coordinates.index]
        express_matrix = express_matrix.T  # Transpose to (genes×cells)
        X = coordinates[['x', 'y']].values.astype(np.float32)
        
        print("Starting SOM training...")
        som = SomNode(X, 10)
        
        plt.figure(figsize=(5, 5))
        som.view()
        plt.title(f'{tissue_type} {dataset} SOM Grid')
        plt.show()
        
        plt.figure(figsize=(5, 5))
        som.viewIniCodebook()
        plt.title(f'{tissue_type} {dataset} Initial Codebook')
        plt.show()
        
        som.reTrain(100)
        
        plt.figure(figsize=(5, 5))
        som.view()
        plt.title(f'{tissue_type} {dataset} Trained SOM')
        plt.show()
        
        print("Running SOMDE...")
        ndf, ninfo = som.mtx(express_matrix)
        nres = som.norm()
        result, SVnum = som.run()
        
        result = result.rename(columns={'g': 'gene', 'qval': 'adjusted_p_value'})
        result = result.drop_duplicates(subset=['gene'])
        result['pred'] = (result['adjusted_p_value'] < 0.05).astype(int)
        result['gene'] = result['gene'].str.replace('-', '.')
        
        result_file = f'../results/{tissue_type}/{dataset}/{tissue_type}_{dataset}_somde_results_processed.csv'
        result.to_csv(result_file, index=False)
        print(f"Analysis completed, results saved to: {result_file}")
        
    except Exception as e:
        print(f"Error processing dataset {dataset}: {str(e)}")
        continue