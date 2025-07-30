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
    
    try:
        adata = sc.read_h5ad(f"../data/CRC/ST-{dataset}/processed_adata.h5ad")
        adata.obs["x_array"] = adata.obs["array_row"]
        adata.obs["y_array"] = adata.obs["array_col"]
        adata.obs["x_pixel"] = pd.Series(adata.obsm['spatial'][:, 1], index=adata.obs.index)
        adata.obs["y_pixel"] = pd.Series(adata.obsm['spatial'][:, 0], index=adata.obs.index)

        x_array = adata.obs["x_array"].tolist()
        y_array = adata.obs["y_array"].tolist()
        x_pixel = adata.obs["x_pixel"].tolist()
        y_pixel = adata.obs["y_pixel"].tolist()

        adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_counts=10)
        spg.prefilter_genes(adata, min_cells=5)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

        s = 1
        b = 49
        adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, beta=b, alpha=s, histology=False)

        p = 0.5
        l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

        n_clusters = 5

        r_seed = t_seed = n_seed = 100

        res = spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

        clf = spg.SpaGCN()
        clf.set_l(l)
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
        y_pred, prob = clf.predict()
        adata.obs["pred"] = y_pred
        adata.obs["pred"] = adata.obs["pred"].astype('category')

        adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
        adata.obs["refined_pred"] = refined_pred
        adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')

        plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
        
        domains="pred"
        num_celltype=len(adata.obs[domains].unique())
        adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
        ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
        ax.set_aspect('equal', 'box')
        ax.axes.invert_yaxis()
        plt.show()

        domains="refined_pred"
        num_celltype=len(adata.obs[domains].unique())
        adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
        ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
        ax.set_aspect('equal', 'box')
        ax.axes.invert_yaxis()
        plt.show()

        raw = sc.read_h5ad(f"../data/CRC/ST-{dataset}/processed_adata.h5ad")
        raw.var_names_make_unique()
        sc.pp.filter_cells(raw, min_counts=10)
        spg.prefilter_genes(raw, min_cells=5)
        raw.obs["x_array"]=raw.obs["array_row"]
        raw.obs["y_array"]=raw.obs["array_col"]
        raw.obs["x_pixel"] = pd.Series(raw.obsm['spatial'][:, 1], index=raw.obs.index)
        raw.obs["y_pixel"] = pd.Series(raw.obsm['spatial'][:, 0], index=raw.obs.index)
        raw.obs["pred"] = adata.obs["pred"].astype('category')
        raw.X = (raw.X.A if issparse(raw.X) else raw.X)
        raw.raw = raw
        sc.pp.log1p(raw)

        min_in_group_fraction = 0.5
        min_in_out_group_ratio = 0.5
        min_fold_change = 1

        adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), np.quantile(adj_2d[adj_2d != 0], q=0.1)

        de_genes_info_dict = {}
        filter_ed_info_dict = {}

        for target in range(n_clusters):
            r = spg.search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=x_array, y=y_array, pred=adata.obs["pred"].tolist(), start=start, end=end, num_min=10, num_max=14, max_run=100)

            nbr_domians = spg.find_neighbor_clusters(target_cluster=target, cell_id=raw.obs.index.tolist(), x=raw.obs["x_array"].tolist(), y=raw.obs["y_array"].tolist(), pred=raw.obs["pred"].tolist(), radius=r, ratio=1/2)
            nbr_domians = nbr_domians[0:3]

            de_genes_info = spg.rank_genes_groups(input_adata=raw, target_cluster=target, nbr_list=nbr_domians, label_col="pred", adj_nbr=True, log=True)
            de_genes_info["target_domain"] = target
            de_genes_info["neighbors"] = str(nbr_domians)
            de_genes_info_dict[target] = de_genes_info

            filtered_info = de_genes_info[(de_genes_info["pvals_adj"] < 0.05) & (de_genes_info["in_out_group_ratio"] > min_in_out_group_ratio) & (de_genes_info["in_group_fraction"] > min_in_group_fraction) & (de_genes_info["fold_change"] > min_fold_change)]
            filtered_info = filtered_info.sort_values(by="in_group_fraction", ascending=False)
            filtered_info["target_domain"] = target
            filtered_info["neighbors"] = str(nbr_domians)
            filter_ed_info_dict[target] = filtered_info

        min_p_values_dict = {}
        for gene in de_genes_info_dict[0]['genes'].unique():
            min_p_value = np.inf
            min_target = None
            for target, de_genes_info in de_genes_info_dict.items():
                p_value = de_genes_info.loc[de_genes_info['genes'] == gene, 'pvals_adj'].values
                if len(p_value) > 0 and p_value[0] < min_p_value:
                    min_p_value = p_value[0]
                    min_target = target
            if min_target is not None:
                min_p_values_dict[gene] = de_genes_info_dict[min_target].loc[de_genes_info_dict[min_target]['genes'] == gene].iloc[0]

        final_de_genes_info = pd.DataFrame(min_p_values_dict.values())

        all_filtered_info = pd.concat(filter_ed_info_dict.values())
        final_filtered_info = all_filtered_info.sort_values(by='pvals_adj').drop_duplicates(subset='genes', keep='first')

        final_de_genes_info = final_de_genes_info.rename(columns={'genes': 'gene', 'pvals_adj': 'adjusted_p_value'})
        final_filtered_info = final_filtered_info.rename(columns={'genes': 'gene', 'pvals_adj': 'adjusted_p_value'})
        final_de_genes_info['pred'] = final_de_genes_info['gene'].isin(final_filtered_info['gene']).astype(int)
        final_de_genes_info['gene'] = final_de_genes_info['gene'].str.replace('-', '.')
        
        result_file = f"../results/{tissue_type}/{dataset}/{tissue_type}_{dataset}_spagcn_results_processed.csv"
        final_de_genes_info.to_csv(result_file, index=False)
        print(f"Results saved to: {result_file}")

    except Exception as e:
        print(f"Error processing {dataset}: {str(e)}")
        continue
