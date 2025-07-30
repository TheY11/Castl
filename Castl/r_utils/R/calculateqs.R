library(dplyr)
library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)
library(patchwork)
library(ggplot2)

#' Calculate Quality Score (QS) metrics for SVG detection methods
#'
#' @param gene_list A dataframe containing ranked gene lists from multiple methods and samples.
#'                 Required columns: gene, sample, method, rank, pred (prediction status)
#' @param sample_ids Character vector of sample IDs to include in analysis
#' @param colors Optional vector of colors for each method (will use default if NULL)
#' @param top_k_values Numeric vector of top K values to evaluate (default: seq(1000, 6000, by=1000))
#' @param org_db Organism database for GO enrichment (default: org.Hs.eg.db)
#' @param ont Ontology for GO enrichment (default: "BP")
#' @param method_labels Optional named vector of display names for methods
#'
#' @return A list containing:
#'         - metrics_df: Long-form dataframe with all calculated metrics
#'         - key_metrics: Wide-form summary table
#'         - plots: List of ggplot objects for visualization
#'         - method_colors: Colors used for each method
#'         - method_labels: Labels used for each method
#'
#' @importFrom dplyr filter group_by summarise ungroup pull mutate arrange distinct n
#' @importFrom tidyr pivot_wider
#' @importFrom clusterProfiler enrichGO
#' @importFrom ggplot2 ggplot aes geom_line geom_point scale_color_manual scale_shape_manual scale_x_continuous scale_y_continuous labs theme_minimal theme element_line element_blank element_text unit guides guide_legend
#' @importFrom rlang sym
#' @importFrom purrr map_dfr
#' @importFrom tibble tibble
#' @importFrom stats setNames
#' @export
calculate_qs <- function(
    gene_list,
    sample_ids,
    colors = NULL,
    top_k_values = seq(1000, 6000, by = 1000),
    org_db = org.Hs.eg.db,
    ont = "BP",
    method_labels = NULL
) {
  
  if (!is.data.frame(gene_list)) {
    stop("gene_list must be a dataframe")
  }
  
  required_cols <- c("gene", "sample", "method", "rank", "pred")
  missing_cols <- setdiff(required_cols, colnames(gene_list))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
  
  methods <- unique(gene_list$method)
  
  default_colors <- c(
    "#97ce9f", "#3480b8", "#82afda", "#add3e2", "#8dcec8",
    "#c2bdde", "#a791c1", "#ffbe7a", "#fa8878", "#c82423",
    "#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0"
  )
  
  if (is.null(colors)) {
    method_colors <- setNames(
      default_colors[seq_along(methods)],
      methods
    )
  } else {
    if (length(colors) < length(methods)) {
      warning("Not enough colors provided, recycling colors")
      colors <- rep(colors, length.out = length(methods))
    }
    method_colors <- setNames(colors[seq_along(methods)], methods)
  }
  
  if (is.null(method_labels)) {
    method_labels <- setNames(methods, methods)
  } else {
    missing_methods <- setdiff(methods, names(method_labels))
    if (length(missing_methods) > 0) {
      method_labels[missing_methods] <- missing_methods
    }
  }
  
  custom_theme <- function(base_size = 14) {
    ggplot2::theme_minimal(base_size = base_size) +
      ggplot2::theme(
        axis.line = ggplot2::element_line(color = "black", linewidth = 0.5),
        axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.5),
        panel.grid = ggplot2::element_blank(),
        legend.position = "right",
        legend.key.height = ggplot2::unit(0.8, "cm"),
        plot.title = ggplot2::element_text(face = "bold", hjust = 0.5, size = 12),
        legend.title = ggplot2::element_text(size = 12, face = "bold"),
        panel.border = ggplot2::element_blank()
      )
  }
  
  calculate_metrics <- function(top_k) {
    purrr::map_dfr(methods, function(m) {
      df <- dplyr::filter(gene_list, method == m, pred == 1)
      top_genes <- df %>% 
        dplyr::filter(rank <= top_k) %>% 
        dplyr::pull(gene) %>% 
        unique()
      
      consistency_df <- df %>% 
        dplyr::filter(gene %in% top_genes) %>% 
        dplyr::group_by(gene) %>% 
        dplyr::summarise(n_samples = dplyr::n()) %>% 
        dplyr::ungroup()
      
      consistency <- if (nrow(consistency_df) > 0) {
        min(mean(consistency_df$n_samples) / length(sample_ids), 1)
      } else {
        0
      }
      
      ego <- suppressMessages(
        clusterProfiler::enrichGO(
          top_genes,
          universe = unique(gene_list$gene),
          OrgDb = org_db,
          keyType = "SYMBOL",
          ont = ont,
          pAdjustMethod = "BH",
          pvalueCutoff = 0.05
        )
      )
      
      functional_specificity <- if (is.null(ego) || nrow(ego) == 0) {
        0
      } else {
        covered_genes <- unique(unlist(strsplit(ego$geneID, "/")))
        length(covered_genes) / length(top_genes)
      }
      
      tibble::tibble(
        method = m,
        top_k = top_k,
        consistency = consistency,
        functional_specificity = functional_specificity,
        quality_score = consistency * functional_specificity
      )
    }) %>% 
      dplyr::mutate(method = factor(method, levels = methods))
  }
  
  metrics_df <- purrr::map_dfr(top_k_values, calculate_metrics)
  
  key_metrics <- metrics_df %>%
    tidyr::pivot_wider(
      names_from = top_k, 
      values_from = c(consistency, functional_specificity, quality_score)
    ) %>%
    dplyr::arrange(method) %>%
    dplyr::mutate(method = method_labels[method])
  
  create_plot <- function(y_var, title) {
    y_sym <- rlang::sym(y_var)
    
    ggplot2::ggplot(metrics_df, ggplot2::aes(top_k, !!y_sym, group = method)) +
      ggplot2::geom_line(ggplot2::aes(color = method), linewidth = 1) +
      ggplot2::geom_point(ggplot2::aes(color = method, shape = method), size = 3) +
      ggplot2::scale_color_manual(
        values = method_colors, 
        labels = method_labels,
        name = "Method"
      ) +
      ggplot2::scale_shape_manual(
        values = seq_along(methods),
        labels = method_labels,
        name = "Method"
      ) +
      ggplot2::scale_x_continuous(
        breaks = top_k_values,
        labels = top_k_values,
        name = "Number of Top-ranked Genes"
      ) +
      ggplot2::scale_y_continuous(
        limits = c(0, 1),
        name = dplyr::case_when(
          y_var == "consistency" ~ "Cross-sample Consistency",
          y_var == "functional_specificity" ~ "Functional Specificity",
          y_var == "quality_score" ~ "Composite Quality Score"
        )
      ) +
      ggplot2::labs(title = title) +
      custom_theme() +
      ggplot2::guides(
        color = ggplot2::guide_legend(ncol = 1),
        shape = ggplot2::guide_legend(ncol = 1)
      )
  }
  
  plots <- list(
    consistency = create_plot("consistency", "Detection Consistency"),
    functional = create_plot("functional_specificity", "Functional Specificity"),
    quality = create_plot("quality_score", "Quality Score")
  )
  
  list(
    metrics_df = metrics_df,
    key_metrics = key_metrics,
    plots = plots,
    method_colors = method_colors,
    method_labels = method_labels
  )
}

#' Load and preprocess SVG detection results from multiple methods
#'
#' This function reads CSV files containing ranked gene lists from different SVG detection methods,
#' standardizes their formats, and combines them into a single dataframe.
#'
#' @param sample_ids Character vector of sample IDs.
#' @param methods Character vector of method names.
#' @return A tibble with columns: `gene`, `sample`, `method`, `rank`, `pred`, and method-specific scores.
#' @export
load_data <- function(sample_ids, methods) {
  
  select <- dplyr::select
  arrange <- dplyr::arrange
  filter <- dplyr::filter
  mutate <- dplyr::mutate
  left_join <- dplyr::left_join
  
  all_data <- purrr::map_dfr(sample_ids, function(sid) {
    purrr::map_dfr(methods, function(m) {
      fname <- file.path("../results/DLPFC/", 
                         sid,
                         paste0("DLPFC_", sid, "_", m, "_results_processed.csv"))
      if (file.exists(fname)) {
        df <- read.csv(fname, stringsAsFactors = FALSE) %>% 
          tibble::as_tibble() %>% 
          mutate(sample = sid, method = m)
        
        if (m == "pvalues_aggregation") {
          if (!"adjusted_p_value" %in% colnames(df)) {
            if ("pvalue" %in% colnames(df)) {
              df <- df %>% mutate(adjusted_p_value = pvalue)
            } else if ("pval" %in% colnames(df)) {
              df <- df %>% mutate(adjusted_p_value = pval)
            } else {
              stop("pvalues_aggregation file missing p-value column")
            }
          }
        }
        return(df)
      }
    })
  }) %>% 
    mutate(gene = stringr::str_remove(gene, "\\..*$"))
  
  pvalues_adj <- all_data %>% 
    filter(method == "pvalues_aggregation", pred == 1) %>% 
    select(gene, sample, adjusted_p_value) %>% 
    distinct()
  
  purrr::map_dfr(methods, function(m) {
    df <- all_data %>% 
      filter(method == m, pred == 1)
    
    if (m == "stabl_aggregation") {
      df <- df %>% 
        left_join(pvalues_adj, by = c("gene", "sample"), 
                  relationship = "many-to-many") %>% 
        arrange(desc(frequency), 
                if ("adjusted_p_value" %in% colnames(.)) adjusted_p_value else frequency)
    } else if (m == "rank_aggregation") {
      df <- arrange(df, desc(score))
    } else if (m %in% c("spatialde", "spark", "sparkx", "somde", "spagcn", "spanve", "heartsvg")) {
      if ("adjusted_p_value" %in% colnames(df)) {
        df <- arrange(df, adjusted_p_value)
      } else if ("pvalue" %in% colnames(df)) {
        df <- arrange(df, pvalue)
      } else {
        df <- arrange(df, desc(frequency))
      }
    } else if (m == "pvalues_aggregation") {
      df <- arrange(df, adjusted_p_value)
    }
    
    df %>% mutate(rank = row_number())
  }) %>% 
    mutate(method = factor(method, levels = methods))
}