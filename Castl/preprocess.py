import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from typing import Optional, Union

def data_preprocess(
    adata: sc.AnnData,
    adata_path: Optional[str] = None,
    counts_path: Optional[str] = None,
    coords_path: Optional[str] = None,
    min_counts: int = 10,
    min_cells: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, sc.AnnData]:
    
    adata.var_names_make_unique()
    print(f"Initial shape: {adata.shape}")
    
    mito_genes = adata.var_names.str.startswith('MT-')
    if sum(mito_genes) == 0:
        mito_genes = adata.var_names.str.startswith('mt-')
    
    adata = adata[:, ~mito_genes]
    print(f"Shape after removing mitochondrial genes: {adata.shape}")
    
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Shape after quality filtering: {adata.shape}")
    
    if adata_path is not None:
        adata.write(adata_path)
        print(f"Saved processed AnnData to: {adata_path}")
    
    express_matrix = pd.DataFrame(
        adata.X.toarray() if sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    coordinates = pd.DataFrame(
        adata.obsm['spatial'],
        index=adata.obs_names,
        columns=['x', 'y']
    )
    
    coordinates['total_counts'] = express_matrix.sum(1)
    express_matrix = express_matrix.loc[coordinates.index]
    
    if counts_path is not None:
        express_matrix.to_csv(counts_path)
        print(f"Saved expression matrix to: {counts_path}")
    
    if coords_path is not None:
        coordinates.to_csv(coords_path)
        print(f"Saved coordinates to: {coords_path}")
    
    return express_matrix, coordinates, adata