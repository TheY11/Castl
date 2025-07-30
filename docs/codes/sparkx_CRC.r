rm(list = ls())

library(SPARK)
library(tibble)
library(dplyr)

datasets <- c('colon2')

for (dataset in datasets) {
  print(paste("\nProcessing dataset:", dataset))
  
  tissue_type <- ifelse(grepl("colon", dataset), "CRC", "Liver")
  
  counts_matrix_path <- file.path("..", "data", "CRC", paste0("ST-", dataset), 
                                  paste0("counts_matrix.csv"))
  col_data_path <- file.path("..", "data", "CRC", paste0("ST-", dataset), 
                             paste0("coordinates.csv"))
  
  output_csv_path <- file.path("..", "results", tissue_type, dataset, 
                               paste0(tissue_type, "_", dataset, "_sparkx_results_processed.csv"))
  
  if (!file.exists(counts_matrix_path)) {
    print(paste("Warning: Counts matrix file not found for", dataset))
    next
  }
  
  if (!file.exists(col_data_path)) {
    print(paste("Warning: Coldata file not found for", dataset))
    next
  }
  
  print(paste("Loading data for", dataset))
  counts <- read.csv(counts_matrix_path, header = TRUE, row.names = 1)
  counts <- t(counts)
  counts <- as.matrix(counts)
  
  info <- read.csv(col_data_path, header = TRUE, row.names = 1)
  info <- info[, 1:2]
  
  print(paste("Data loaded for", dataset, "- Dimensions:", dim(counts)))
  
  print(paste("Running SPARK-X for", dataset))
  sparkx <- sparkx(counts, info, numCores = 4, option = "mixture")
  
  print(paste("Extracting results for", dataset))
  results <- sparkx$res_mtest
  results <- rownames_to_column(results, var = "gene")
  results <- results %>% rename(adjusted_p_value = adjustedPval)
  results <- results %>% mutate(pred = ifelse(adjusted_p_value < 0.05, 1, 0))
  results <- results %>% mutate(gene = gsub("-", ".", gene))
  
  print(paste("Saving results to", output_csv_path))
  write.csv(results, file = output_csv_path, row.names = FALSE)
  
  print(paste("Completed SPARK-X analysis for", dataset))
}

print("Finished processing all datasets with SPARK-X")