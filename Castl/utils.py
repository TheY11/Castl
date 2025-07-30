import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import roc_curve, auc
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Set, Dict, Optional, Tuple, Union, Literal

def plot_gene(
    adata,
    gene_df: pd.DataFrame,
    x_col: str = 'x_pixel',
    y_col: str = 'y_pixel',
    gene_col: str = 'gene',
    sort_col: str = 'adjusted_p_value',
    ascending: bool = True,
    top_n: int = 12,
    spotsize: int = 8,
    cmap: Union[Literal['pink_green', 'white_pink'], clr.Colormap] = 'pink_green',
    figsize: tuple = (20, 10),
    invert_yaxis: bool = False,
    invert_xaxis: bool = False
) -> None:
    
    genes_per_row = 6
    
    if isinstance(cmap, str):
        if cmap == 'pink_green':
            cmap = clr.LinearSegmentedColormap.from_list(
                'pink_green', ['#3AB370', "#EAE7CC", "#FD1593"], N=256
            )
        elif cmap == 'white_pink':
            nodes = [0.0, 0.05, 0.3, 1.0]
            cmap = clr.LinearSegmentedColormap.from_list(
                'white_pink', list(zip(nodes, ["#EAE7CC","#EAE7CC","#FD1593","#FD1593"]))
            )
        else:
            raise ValueError("cmap must be either 'pink_green', 'white_pink' or a custom colormap")
    
    sorted_genes = gene_df.sort_values(sort_col, ascending=ascending)[gene_col].values[:top_n]
    
    num_rows = (top_n + genes_per_row - 1) // genes_per_row
    num_cols = min(genes_per_row, top_n)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows > 1 or num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, gene in enumerate(sorted_genes):
        ax = axes[i]
        
        original_gene = gene.replace(".", "-")
        
        if original_gene in adata.var_names:
            gene_name = original_gene
        elif gene in adata.var_names:
            gene_name = gene
        else:
            print(f"Warning: Gene '{gene}' (tried as '{original_gene}') not found in adata.var_names. Skipping...")
            ax.axis('off')
            continue
            
        try:
            adata.obs["exp"] = adata[:, gene_name].X.toarray().flatten()
            
            sns.scatterplot(
                x=x_col, 
                y=y_col, 
                hue="exp", 
                palette=cmap, 
                data=adata.obs, 
                ax=ax, 
                s=spotsize
            )
            
            if invert_yaxis:
                ax.invert_yaxis()
            if invert_xaxis:
                ax.invert_xaxis()
            
            ax.set_title(f"{gene_name}", fontsize=16)
            ax.set_aspect('equal', 'box')
            ax.get_legend().remove()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(False)
        except Exception as e:
            print(f"Error plotting gene {gene}: {str(e)}")
            ax.axis('off')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()

def read_gmt(file_path: str) -> Dict[str, Dict[str, List[str]]]:
    genesets = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            geneset_name = parts[0]
            description = parts[1]
            genes = parts[2:]
            genesets[geneset_name] = {
                'description': description,
                'genes': genes
            }
    return genesets

def get_geneset(gmt_file: str) -> Set[str]:
    try:
        genesets = read_gmt(gmt_file)
        true_genes = set()
        for geneset in genesets.values():
            true_genes.update([gene.replace('-', '.') for gene in geneset['genes']])
        print(f"Loaded {len(true_genes)} genes from GMT file")
        return true_genes
    except Exception as e:
        print(f"Error processing GMT file: {str(e)}")
        return set()

def plot_ROC(
    true_labels: Set[str],
    methods: Dict[str, Tuple[pd.DataFrame, str]],
    colors: Optional[List[str]] = None,
    gene_col: str = 'gene',
    figsize: tuple = (6, 5.2),
    fontsize: int = 18,
    legend_fontsize: int = 10,
    linewidth: int = 3
) -> None:

    plt.figure(figsize=figsize)
    
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    if colors is None:
        colors = default_colors[:len(methods)]
    elif len(colors) < len(methods):
        print(f"Warning: Not enough colors provided. Using default colors for remaining methods.")
        colors = colors + default_colors[len(colors):len(methods)]
    
    for i, ((method_name, (df, score_col)), color) in enumerate(zip(methods.items(), colors)):
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"Warning: {method_name} data is not a valid DataFrame. Skipping.")
            continue
            
        if gene_col not in df.columns:
            print(f"Warning: {method_name} missing gene column '{gene_col}'. Available columns: {df.columns.tolist()}")
            continue
        if score_col not in df.columns:
            print(f"Warning: {method_name} missing score column '{score_col}'. Available columns: {df.columns.tolist()}")
            continue
            
        y_true = df[gene_col].isin(true_labels).astype(int)
        y_score = df[score_col].fillna(0)
        
        if 'p_value' in score_col.lower() or 'pvalue' in score_col.lower():
            y_score = 1 - y_score
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            if not np.isnan(roc_auc):
                plt.plot(fpr, tpr, color=color, linewidth=linewidth,
                         label=f'{method_name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Error processing {method_name}: {str(e)}")
    
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right', fontsize=legend_fontsize)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    plt.tight_layout()