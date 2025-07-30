library(dplyr)
library(TissueEnrich)
library(ggplot2)

#' Perform Tissue Enrichment Analysis (TEA) on SVG genes
#'
#' @param gene_list A dataframe containing gene predictions with columns:
#'                  - gene: gene symbols
#'                  - pred: binary prediction status (1 = SVG, 0 = non-SVG)
#' @param organism Organism name (default: "Mus Musculus")
#' @param top_n Number of top enriched tissues to display (default: 8)
#' @param colors Optional vector of colors for tissues (will use default if NULL)
#' @param fig_size Optional vector specifying figure size (width, height) in inches
#'
#' @return A list containing:
#'         - enrichment_results: Full enrichment results dataframe
#'         - top_tissues: Vector of top enriched tissues
#'         - plot: ggplot object of the enrichment plot
#'         - tissue_colors: Colors used for each tissue
#'
#' @importFrom TissueEnrich teEnrichment
#' @importFrom ggplot2 ggplot aes geom_bar geom_text labs theme_bw theme element_text ylim scale_fill_manual element_blank element_line
#' @importFrom dplyr filter pull arrange desc mutate
#' @importFrom stats setNames
#' @importFrom SummarizedExperiment assay rowData colData
#' @export
perform_tea <- function(
    gene_list,
    organism = "Mus Musculus",
    top_n = 8,
    colors = NULL,
    fig_size = NULL
) {
  if (!is.data.frame(gene_list)) {
    stop("gene_list must be a dataframe")
  }
  
  required_cols <- c("gene", "pred")
  missing_cols <- setdiff(required_cols, colnames(gene_list))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
  
  svg_genes <- gene_list %>% 
    dplyr::filter(pred == 1) %>% 
    dplyr::pull(gene) %>% 
    unique()
  
  if (length(svg_genes) == 0) {
    stop("No SVG genes found in the input data")
  }
  
  gs <- GeneSet(
    geneIds = svg_genes,
    organism = organism,
    geneIdType = SymbolIdentifier()
  )
  
  output <- teEnrichment(inputGenes = gs)
  
  seEnrichmentOutput <- output[[1]]
  enrichmentOutput <- stats::setNames(
    data.frame(
      SummarizedExperiment::assay(seEnrichmentOutput),
      row.names = SummarizedExperiment::rowData(seEnrichmentOutput)[, 1]
    ),
    SummarizedExperiment::colData(seEnrichmentOutput)[, 1]
  )
  enrichmentOutput$Tissue <- row.names(enrichmentOutput)
  
  top_tissues <- enrichmentOutput %>%
    dplyr::arrange(dplyr::desc(Log10PValue)) %>%
    head(top_n) %>%
    dplyr::pull(Tissue)
  
  filtered_results <- enrichmentOutput %>%
    dplyr::filter(Tissue %in% top_tissues) %>%
    dplyr::mutate(Tissue = factor(Tissue, levels = top_tissues))
  
  if (is.null(colors)) {
    colors <- c('#ea100c', '#ffb92a', '#feeb51', '#9bca3e', '#3abbc9', 
                '#1bb6f4', '#7a6fca', '#e667cd', '#a6cee3', '#1f78b4',
                '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928')
    colors <- colors[1:top_n]
    if (top_n > length(colors)) {
      warning("Default palette has 20 colors, recycling colors for top_n > 20")
      colors <- rep(colors, length.out = top_n)
    }
  }
  names(colors) <- top_tissues
  
  tea_plot <- ggplot2::ggplot(
    filtered_results,
    ggplot2::aes(x = Tissue, y = Log10PValue, fill = Tissue)
  ) +
    ggplot2::geom_bar(stat = 'identity') +
    ggplot2::geom_text(
      ggplot2::aes(label = round(Log10PValue, 2)),
      vjust = -0.5, 
      size = 3, 
      color = "black"
    ) +
    ggplot2::labs(x = '', y = 'Log10(P-value)') +
    ggplot2::theme_bw() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 10),
      axis.text.y = ggplot2::element_text(size = 10),
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "none",
      panel.border = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(color = "black"),
      axis.line.y = ggplot2::element_line(color = "black")
    ) +
    ggplot2::scale_fill_manual(values = colors) +
    ggplot2::ylim(0, max(filtered_results$Log10PValue) * 1.1)
  
  if (!is.null(fig_size)) {
    if (length(fig_size) != 2) {
      warning("fig_size should be a vector of length 2 (width, height). Using default size.")
    } else {
      tea_plot <- tea_plot + ggplot2::theme(
        plot.margin = ggplot2::unit(c(1, 1, 1, 1), "cm"),
        plot.background = ggplot2::element_rect(fill = "white")
      )
    }
  }
  
  list(
    enrichment_results = enrichmentOutput,
    top_tissues = top_tissues,
    plot = tea_plot,
    tissue_colors = colors
  )
}