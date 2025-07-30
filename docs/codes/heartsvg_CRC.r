rm(list = ls())

library(HEARTSVG)
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
                               paste0(tissue_type, "_", dataset, "_heartsvg_results_processed.csv"))
  
  if (!file.exists(counts_matrix_path)) {
    print(paste("Warning: Counts matrix file not found for", dataset))
    next
  }
  
  if (!file.exists(col_data_path)) {
    print(paste("Warning: Coldata file not found for", dataset))
    next
  }
  
  print(paste("Loading data for", dataset))
  expression_matrix <- read.csv(counts_matrix_path, row.names = 1)
  position_info <- read.csv(col_data_path, row.names = 1)
  
  position_info <- position_info[, 1:2]
  colnames(position_info) <- c("row", "col")
  
  if (nrow(position_info) != nrow(expression_matrix)) {
    stop(paste("Position information and expression matrix have inconsistent row numbers, cannot process", dataset))
  }
  
  expression_position <- cbind(position_info, expression_matrix)
  
  print(paste("Data loaded for", dataset, "- Dimensions:", dim(expression_position)))
  
  print(paste("Running HEARTSVG for", dataset))
  result <- heartsvg(expression_position, scale = TRUE)
  
  print(paste("Processing results for", dataset))
  result <- result %>% rename(adjusted_p_value = p_adj)
  result <- result %>% mutate(pred = ifelse(adjusted_p_value < 0.05, 1, 0))
  result <- result %>% mutate(gene = gsub("-", ".", gene))
  
  print(paste("Saving results to", output_csv_path))
  write.csv(result, output_csv_path, row.names = TRUE)
  
  print(paste("Completed HEARTSVG analysis for", dataset))
}

print("Finished processing all datasets with HEARTSVG")