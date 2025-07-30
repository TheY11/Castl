import os
import numpy as np
import pandas as pd
import NaiveDE
import SpatialDE
import time
import scanpy as sc

datasets = ['colon2']

for dataset in datasets:
    print(f"\nProcessing dataset {dataset}...")
    
    tissue_type = 'CRC' if dataset.startswith('colon') else 'Liver'
    
    adata_path = f"../data/CRC/ST-{dataset}/{tissue_type}_{dataset}_combined_adata.h5ad"
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)

    express_matrix = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    coordinates = adata.obsm['spatial']
    coordinates = pd.DataFrame(coordinates, index=adata.obs_names, columns=['x', 'y'])

    coordinates['total_counts'] = express_matrix.sum(1)
    express_matrix = express_matrix.loc[coordinates.index]
    X = coordinates[['x', 'y']]

    start_time = time.time()
    dfm = NaiveDE.stabilize(express_matrix.T).T
    res = NaiveDE.regress_out(coordinates, dfm.T, 'np.log(total_counts)').T
    
    results = SpatialDE.run(X, res.T)
    
    elapsed_time = time.time() - start_time

    results = results.rename(columns={'g': 'gene', 'qval': 'adjusted_p_value'})
    results = results.drop_duplicates(subset=['gene'])
    results['pred'] = (results['adjusted_p_value'] < 0.05).astype(int)
    results['gene'] = results['gene'].str.replace('-', '.')

    output_path = f"../results/{tissue_type}/{dataset}/{dataset}_stabl/{tissue_type}_{dataset}_spatialde_stabl_processed.csv"
    results.to_csv(output_path, index=False)
    print(f"Dataset {dataset} processed in {elapsed_time:.2f} seconds. Results saved to {output_path}")

print("\nAll datasets processed successfully!")