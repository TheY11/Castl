import pandas as pd
import numpy as np
from typing import List, Optional
from statsmodels.stats.multitest import multipletests
from anndata import AnnData
import scipy.sparse as sp

def create_combined_matrix(
    expression_matrix: pd.DataFrame,
    random_seed: Optional[int] = None
) -> pd.DataFrame:

    if random_seed is not None:
        np.random.seed(random_seed)
    
    expr_genes = expression_matrix.T
    combined_genes = _create_combined_dataframe(expr_genes)
    return combined_genes.T

def create_combined_adata(
    adata: AnnData,
    random_seed: Optional[int] = None
) -> AnnData:

    if random_seed is not None:
        np.random.seed(random_seed)
    
    if sp.issparse(adata.X):
        expr_matrix = pd.DataFrame(
            adata.X.toarray().T,
            index=adata.var_names,
            columns=adata.obs_names
        )
    else:
        expr_matrix = pd.DataFrame(
            adata.X.T,
            index=adata.var_names,
            columns=adata.obs_names
        )
    
    combined_df = _create_combined_dataframe(expr_matrix)
    
    combined_adata = AnnData(
        X=combined_df.T,
        var=pd.DataFrame(index=combined_df.index),
        obs=adata.obs.copy()
    )
    
    combined_adata.uns = adata.uns.copy()
    if 'spatial' in adata.obsm:
        combined_adata.obsm['spatial'] = adata.obsm['spatial'].copy()
    
    return combined_adata

def _create_combined_dataframe(expression_matrix: pd.DataFrame) -> pd.DataFrame:

    shuffled_values = np.random.permutation(expression_matrix.values.flatten())
    shuffled_expression_matrix = shuffled_values.reshape(expression_matrix.shape)
    
    num_genes = expression_matrix.shape[0]
    artificial_gene_names = [f"ArtGene{i+1}" for i in range(num_genes)]
    
    artificial_expression_matrix = pd.DataFrame(
        shuffled_expression_matrix,
        index=artificial_gene_names,
        columns=expression_matrix.columns
    )
    
    return pd.concat([expression_matrix, artificial_expression_matrix])

def rank_agg(
    gene_list: List[pd.DataFrame],
    gene_col: str = 'gene',
    rank_col: Optional[str] = None,
    ascending: bool = True,
    top_percent: float = 0.1
) -> pd.DataFrame:

    all_genes = set()
    for df in gene_list:
        all_genes.update(df[gene_col].unique())
    all_genes = list(all_genes)
    
    scored_dfs = []
    for df in gene_list:
        df = df.copy()
        
        if rank_col is not None:
            df = df.sort_values(rank_col, ascending=ascending)
        
        df['score'] = range(len(df), 0, -1)
        
        scored_dfs.append(
            df.set_index(gene_col)['score']
            .reindex(all_genes)
            .fillna(0)
        )
    
    result = (
        pd.concat(scored_dfs, axis=1)
        .sum(axis=1)
        .reset_index()
        .rename(columns={'index': gene_col, 0: 'score'})
        .sort_values('score', ascending=False)
    )
    
    result['rank'] = result['score'].rank(method='min', ascending=False).astype(int)
    
    top_n = max(1, int(len(result) * top_percent))
    result['pred'] = 0
    result.iloc[:top_n, result.columns.get_loc('pred')] = 1
    
    return result

def pval_agg(
    gene_list: List[pd.DataFrame],
    gene_col: str = 'gene',
    pvalue_col: str = 'pvalue',
    alpha: float = 0.05,
    correction: str = 'fdr_by'
) -> pd.DataFrame:

    all_genes = set()
    fill_na = 1.0
    for df in gene_list:
        all_genes.update(df[gene_col].unique())
    all_genes = list(all_genes)
    
    processed_lists = []
    for i, df in enumerate(gene_list):
        if pvalue_col not in df.columns:
            raise ValueError(f"DataFrame {i} missing p-value column '{pvalue_col}'")
            
        processed = (
            df.set_index(gene_col)[pvalue_col]
            .reindex(all_genes)
            .fillna(fill_na)
            .rename(f'method_{i}')
        )
        processed_lists.append(processed)
    
    cauchy_stats = [np.tan(np.pi * (0.5 - pvals)) for pvals in processed_lists]
    combined_cauchy = np.sum(cauchy_stats, axis=0)
    combined_p = 0.5 - np.arctan(combined_cauchy) / np.pi
    
    _, adj_p, _, _ = multipletests(combined_p, method=correction)
    
    result = pd.DataFrame({
        gene_col: all_genes,
        'combined_p_value': combined_p,
        'adjusted_p_value': adj_p
    })
    
    result = result.sort_values('adjusted_p_value', ascending=True)
    
    result['rank'] = range(1, len(result) + 1)
    
    result['pred'] = (result['adjusted_p_value'] < alpha).astype(int)
    
    return result

def stabl_agg(
    gene_list: List[pd.DataFrame],
    gene_col: str = 'gene',
    pred_col: str = 'pred',
    penalty_factor: float = 0.1,
    plot: bool = False
) -> pd.DataFrame:

    all_genes = set()
    for df in gene_list:
        all_genes.update(df[gene_col].unique())
    all_genes = list(all_genes)
    
    gene_frequency = {gene: 0 for gene in all_genes}
    method_count = len(gene_list)
    
    for df in gene_list:
        svg_genes = df[df[pred_col] == 1][gene_col]
        for gene in svg_genes:
            gene_frequency[gene] += 1
    
    gene_frequency_df = pd.DataFrame({
        gene_col: list(gene_frequency.keys()),
        'frequency': [freq / method_count for freq in gene_frequency.values()]
    })
    
    gene_frequency_df = gene_frequency_df.sort_values('frequency', ascending=False)
    
    gene_frequency_df['rank'] = range(1, len(gene_frequency_df) + 1)
    
    thresholds = np.linspace(0, 1, 100)
    
    fdp_plus_values = []
    artgene_counts = []
    
    for t in thresholds:
        above_threshold = gene_frequency_df[gene_frequency_df['frequency'] > t]
        
        artgene_count = above_threshold[above_threshold[gene_col].str.startswith('ArtGene')].shape[0]
        non_artgene_count = above_threshold[~above_threshold[gene_col].str.startswith('ArtGene')].shape[0]
        
        fdp_plus = (artgene_count + 1) / max(non_artgene_count, 1)
        
        fdp_plus_values.append(fdp_plus)
        artgene_counts.append(artgene_count)
    
    combined_values = [
        fdp_plus + penalty_factor * artgene_count 
        for fdp_plus, artgene_count in zip(fdp_plus_values, artgene_counts)
    ]
    
    min_combined_index = np.argmin(combined_values)
    optimal_threshold = thresholds[min_combined_index]
    
    gene_frequency_df['pred'] = (gene_frequency_df['frequency'] > optimal_threshold).astype(int)
    
    final_selection = gene_frequency_df[gene_frequency_df['pred'] == 1]
    total_svgs = final_selection.shape[0]
    artgene_count = final_selection[final_selection[gene_col].str.startswith('ArtGene')].shape[0]
    non_artgene_count = total_svgs - artgene_count
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Total SVGs selected: {total_svgs}")
    print(f"ArtGene count in selection: {artgene_count}")
    print(f"Non-ArtGene count in selection: {non_artgene_count}")
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fdp_norm = (np.array(fdp_plus_values) - np.min(fdp_plus_values)) / (
                np.max(fdp_plus_values) - np.min(fdp_plus_values))
            combined_norm = (np.array(combined_values) - np.min(combined_values)) / (
                np.max(combined_values) - np.min(combined_values))
            
            plt.figure(figsize=(6, 4))
            plt.plot(thresholds, fdp_norm, label='FDP+', linewidth=2)
            plt.plot(thresholds, combined_norm, label='Adjusted FDP+', linestyle='--', linewidth=2)
            plt.axvline(x=optimal_threshold, color='r', linestyle=':', label=f'Optimal Threshold', linewidth=2)
            plt.xlabel('Frequency Threshold (t)')
            plt.ylabel('FDP+ and Adjusted FDP+')
            plt.title('Stabl Threshold Optimization')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Set plot=False to skip plotting.")
    
    return gene_frequency_df