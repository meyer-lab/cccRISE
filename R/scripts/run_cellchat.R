#' Run CellChat (jinworks/CellChat) on an exported dataset.
#'
#' Usage: Rscript run_cellchat.R <data_dir> <group_by_column> <out_dir>
#' Example: Rscript run_cellchat.R ../data/balf_covid celltype ../output/cellchat

suppressPackageStartupMessages({
  library(CellChat)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[[1]]
group_by <- args[[2]]
out_dir <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("lib_load_data.R")  # run scripts from within R/scripts/ (see run_all.R)

seurat_obj <- load_exported_seurat(data_dir)
seurat_obj <- NormalizeData(seurat_obj)
Idents(seurat_obj) <- seurat_obj[[group_by, drop = TRUE]]

cellchat <- createCellChat(object = seurat_obj, group.by = group_by, assay = "RNA")
cellchat@DB <- CellChatDB.human

cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
cellchat <- computeCommunProb(cellchat, type = "triMean")
cellchat <- filterCommunication(cellchat, min.cells = 10)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)

df_lr <- subsetCommunication(cellchat)
write.csv(df_lr, file.path(out_dir, "lr_communication.csv"), row.names = FALSE)

df_pathway <- subsetCommunication(cellchat, slot.name = "netP")
write.csv(df_pathway, file.path(out_dir, "pathway_communication.csv"), row.names = FALSE)

net_count <- cellchat@net$count
net_weight <- cellchat@net$weight
write.csv(as.data.frame(net_count), file.path(out_dir, "celltype_interaction_count.csv"))
write.csv(as.data.frame(net_weight), file.path(out_dir, "celltype_interaction_weight.csv"))

saveRDS(cellchat, file.path(out_dir, "cellchat_object.rds"))
cat("CellChat done ->", out_dir, "\n")
