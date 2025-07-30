import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scanpy as sc
from somde import SomNode
from somde.util import *

rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

datasets = ['colon2']

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    
    tissue_type = 'CRC' if dataset.startswith('colon') else 'Liver'
    
    adata_path = f"../data/CRC/ST-{dataset}/{tissue_type}_{dataset}_combined_adata.h5ad"
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)

    express_matrix = pd.DataFrame(adata.X.toarray().T,
                                  index=adata.var_names, 
                                  columns=adata.obs_names)

    coordinates = adata.obsm['spatial']
    coordinates = pd.DataFrame(coordinates, index=adata.obs_names, columns=['x', 'y'])
    coordinates['total_counts'] = express_matrix.sum(axis=1)
    X = coordinates[['x', 'y']].values.astype(np.float32)

    print("Training SOM...")
    som = SomNode(X, 10)
    
    print("Running matrix transformation and analysis...")
    ndf, ninfo = som.mtx(express_matrix)
    nres = som.norm()
    result, SVnum = som.run()

    result = result.rename(columns={'g': 'gene', 'qval': 'adjusted_p_value'})
    result = result.drop_duplicates(subset=['gene'])
    result['pred'] = (result['adjusted_p_value'] < 0.05).astype(int)
    result['gene'] = result['gene'].str.replace('-', '.')

    output_path = f"../results/{tissue_type}/{dataset}/{dataset}_stabl/{tissue_type}_{dataset}_somde_stabl_processed.csv"
    result.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

print("\nAll datasets processed successfully!")