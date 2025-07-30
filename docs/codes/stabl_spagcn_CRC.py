import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2

datasets = ['colon2']

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    
    tissue_type = 'CRC' if dataset.startswith('colon') else 'Liver'
    
    adata_path = f"../data/CRC/ST-{dataset}/{tissue_type}_{dataset}_combined_adata.h5ad"
    adata = sc.read_h5ad(adata_path)
    
    adata.obs["x_array"] = adata.obsm['spatial'][:, 0]
    adata.obs["y_array"] = adata.obsm['spatial'][:, 1]
    adata.obs["x_pixel"] = adata.obsm['spatial'][:, 0]
    adata.obs["y_pixel"] = adata.obsm['spatial'][:, 1]

    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=10)
    spg.prefilter_genes(adata, min_cells=5)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    s = 1
    b = 49
    adj = spg.calculate_adj_matrix(
        x=adata.obs["x_pixel"], 
        y=adata.obs["y_pixel"], 
        x_pixel=adata.obs["x_pixel"], 
        y_pixel=adata.obs["y_pixel"], 
        beta=b, 
        alpha=s, 
        histology=False
    )

    p = 0.5
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    n_clusters = 5

    r_seed = t_seed = n_seed = 100
    res = spg.search_res(
        adata, adj, l, n_clusters, 
        start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, 
        r_seed=r_seed, t_seed=t_seed, n_seed=n_seed
    )

    clf = spg.SpaGCN()
    clf.set_l(l)
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    clf.train(
        adata, adj, init_spa=True, init="louvain", res=res, 
        tol=5e-3, lr=0.05, max_epochs=200
    )
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred.astype('category')

    raw = sc.read_h5ad(adata_path)
    raw.var_names_make_unique()
    sc.pp.filter_cells(raw, min_counts=10)
    spg.prefilter_genes(raw, min_cells=5)
    raw.obs = adata.obs.copy()
    raw.X = (raw.X.A if issparse(raw.X) else raw.X)
    raw.raw = raw
    sc.pp.log1p(raw)

    min_in_group_fraction = 0.8
    min_in_out_group_ratio = 1
    min_fold_change = 1.5
    adj_2d = spg.calculate_adj_matrix(x=raw.obs["x_array"], y=raw.obs["y_array"], histology=False)
    start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), np.quantile(adj_2d[adj_2d != 0], q=0.1)

    de_genes_info_dict = {}
    filtered_info_dict = {}

    for target in range(n_clusters):
        r = spg.search_radius(
            target_cluster=target, 
            cell_id=raw.obs.index.tolist(), 
            x=raw.obs["x_array"], 
            y=raw.obs["y_array"], 
            pred=raw.obs["pred"], 
            start=start, end=end, num_min=10, num_max=14, max_run=100
        )
        
        nbr_domians = spg.find_neighbor_clusters(
            target_cluster=target, 
            cell_id=raw.obs.index.tolist(), 
            x=raw.obs["x_array"], 
            y=raw.obs["y_array"], 
            pred=raw.obs["pred"], 
            radius=r, 
            ratio=1/2
        )[:3]
        
        de_genes_info = spg.rank_genes_groups(
            input_adata=raw, 
            target_cluster=target, 
            nbr_list=nbr_domians, 
            label_col="pred", 
            adj_nbr=True, 
            log=True
        )
        de_genes_info["target_domain"] = target
        de_genes_info["neighbors"] = str(nbr_domians)
        de_genes_info_dict[target] = de_genes_info
        
        filtered_info = de_genes_info[
            (de_genes_info["pvals_adj"] < 0.05) &
            (de_genes_info["in_out_group_ratio"] > min_in_out_group_ratio) &
            (de_genes_info["in_group_fraction"] > min_in_group_fraction) &
            (de_genes_info["fold_change"] > min_fold_change)
        ].sort_values(by="in_group_fraction", ascending=False)
        filtered_info["target_domain"] = target
        filtered_info["neighbors"] = str(nbr_domians)
        filtered_info_dict[target] = filtered_info

    final_de_genes_info = pd.concat(de_genes_info_dict.values())
    final_de_genes_info = final_de_genes_info.sort_values(by="pvals_adj").drop_duplicates(subset="genes", keep="first")
    final_de_genes_info = final_de_genes_info.rename(
        columns={"genes": "gene", "pvals_adj": "adjusted_p_value"}
    )
    final_de_genes_info["pred"] = final_de_genes_info["gene"].isin(
        pd.concat(filtered_info_dict.values())["gene"]
    ).astype(int)
    final_de_genes_info["gene"] = final_de_genes_info["gene"].str.replace("-", ".")

    output_path = f".../results/{tissue_type}/{dataset}/{dataset}_stabl/{tissue_type}_{dataset}_spagcn_stabl_processed.csv"
    final_de_genes_info.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

print("\nAll datasets processed successfully!")