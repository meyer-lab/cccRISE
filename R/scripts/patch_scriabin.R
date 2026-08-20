#' Runtime patch for scriabin::GenerateInteractome (BlishLab/scriabin@HEAD).
#'
#' Source this after `library(scriabin)` and before calling
#' GenerateInteractome(). Unlike the STACCato patch (apply_patches.R), this
#' one is applied in-memory per-session via assignInNamespace rather than to
#' the installed package files on disk, because the fix is a function-body
#' change rather than a metadata/NAMESPACE edit.
#'
#' Two independent upstream bugs are fixed, both confirmed by inspecting the
#' installed package with `dump("GenerateInteractome", ...)`:
#'
#' 1. `slot = slot.use` is passed to SeuratObject::GetAssayData(), but the
#'    `slot` argument was removed (defunct) in SeuratObject >= 5.0 in favor
#'    of `layer`. Every GenerateInteractome() call errors immediately.
#'
#' 2. The `database = "<built-in name>"` path (e.g. "CellChatDB") indexes
#'    inst/lr_resources.rds directly by database name, but the bundled
#'    resource is actually nested one level deeper by species
#'    (`lr_resources.rds$human$CellChatDB`), so every non-"custom" database
#'    argument fails with "Database must be one of: ...". We keep using
#'    `database = "custom"` (which already works, see run_scriabin.R) so this
#'    fix is included for completeness but not exercised by our scripts.
#'
#' This patched copy also drops the upstream `Sys.sleep(0.5)` per source cell
#' (a progress-bar throttle with no functional effect on the result), since
#' it turns even a modest send_cells x rec_cells run into a multi-minute wait.

stopifnot(requireNamespace("scriabin", quietly = TRUE))

patched_GenerateInteractome <- function(
  seu, ranked_genes, cluster_results = NULL,
  send_cluster = NULL, rec_cluster = NULL, ident.label = NULL,
  use_clusters = FALSE, send_cells = NULL, rec_cells = NULL,
  cell.type.calls = "celltype.l2", assay.use = "SCT", slot.use = "data",
  group.by = "time.orig", database = "OmniPath"
) {
  `%notin%` <- scriabin::`%notin%`
  if (use_clusters) {
    if (!is.null(send_cells) | !is.null(rec_cells)) {
      stop("Cell names are supplied both through cluster results and send_cells/rec_cells.\n           If you wish to supply cell names explicitly, set use_clusters = F")
    }
    send_nb <- names(cluster_results$SCluster)[cluster_results$SCluster %in% send_cluster]
    rec_nb <- names(cluster_results$RCluster)[cluster_results$RCluster %in% rec_cluster]
    send_cells <- colnames(seu)[seu$bins %in% send_nb & (seu@meta.data[, group.by] %in% ident.label)]
    rec_cells <- colnames(seu)[seu$bins %in% rec_nb & (seu@meta.data[, group.by] %in% ident.label)]
  } else if (is.null(send_cells) | is.null(rec_cells)) {
    stop("Must supply cells to analyze either in cluster results, or as vectors send_cells and rec_cells")
  }

  if (database == "custom") {
    pairs <- data.frame(ligands = ligands, recepts = recepts)  # from caller's environment, as upstream intends
  } else {
    all <- readRDS(system.file(package = "scriabin", "lr_resources.rds"))$human  # fix: species-nested resource
    if (database %notin% names(all)) {
      stop("Database must be one of: ", paste(names(all), collapse = ", "))
    }
    pairs <- as.data.frame(all[[database]][, c("source_genesymbol", "target_genesymbol")] |>
      dplyr::mutate_all(as.character))
  }
  colnames(pairs) <- c("ligand", "receptor")

  cell.exprs <- SeuratObject::GetAssayData(seu, assay = assay.use, layer = slot.use)  # fix: slot -> layer
  cell.exprs <- cell.exprs[rownames(seu) %in% c(unique(pairs$ligand), unique(pairs$receptor)),
                            c(send_cells, rec_cells)]
  cell.exprs <- cell.exprs[Matrix::rowSums(cell.exprs) > 0, ]

  sources <- send_cells
  targets <- rec_cells
  message(paste0(length(sources), " sources"))
  message(paste0(length(targets), " targets"))
  message("Generating Interactome")
  pb <- txtProgressBar(min = 0, max = length(sources), initial = 0, style = 3)
  connectome <- data.frame()
  for (i in seq_along(sources)) {
    temp <- data.frame()
    for (j in seq_along(targets)) {
      cell.exprs_use <- cell.exprs[, c(sources[i], targets[j])]
      cell.exprs_use <- as.data.frame(cell.exprs_use[Matrix::rowSums(cell.exprs_use) > 0, ]) |>
        tibble::rownames_to_column("gene")
      pairs_use <- pairs |>
        dplyr::filter(ligand %in% cell.exprs_use$gene) |>
        dplyr::filter(receptor %in% cell.exprs_use$gene)
      if (nrow(cell.exprs_use) > 0 && sources[i] != targets[j] && nrow(pairs_use) > 0) {
        lig_i <- pairs_use[, "ligand"]
        recept_j <- pairs_use[, "receptor"]
        ligands.df <- data.frame(ligands = lig_i, id = seq_along(lig_i))
        recepts.df <- data.frame(recepts = recept_j, id = seq_along(recept_j))
        cell.exprs_use.rec <- merge(recepts.df, cell.exprs_use[, c("gene", rec_cells[j])],
                                     by.x = "recepts", by.y = "gene", all.x = TRUE)
        cell.exprs_use.rec <- cell.exprs_use.rec[order(cell.exprs_use.rec$id), ]
        cell.exprs_use.lig <- merge(ligands.df, cell.exprs_use[, c("gene", send_cells[i])],
                                     by.x = "ligands", by.y = "gene", all.x = TRUE)
        cell.exprs_use.lig <- cell.exprs_use.lig[order(cell.exprs_use.lig$id), ]
        lig_geneset <- lig_i %in% names(ranked_genes[[sources[i]]])
        rec_geneset <- recept_j %in% names(ranked_genes[[targets[j]]])
        vector <- data.frame(
          source = sources[i], receiver = targets[j], ligand = lig_i, receptor = recept_j,
          pair = paste(lig_i, recept_j, sep = " - "),
          ligand.expression = cell.exprs_use.lig[, sources[i]],
          recept.expression = cell.exprs_use.rec[, targets[j]],
          lig_geneset = lig_geneset, rec_geneset = rec_geneset
        )
        temp <- rbind(temp, vector)
      }
    }
    connectome <- rbind(connectome, temp)
    setTxtProgressBar(pb, i)
  }
  close(pb)

  connectome$ligand.expression <- log1p(connectome$ligand.expression)
  connectome$recept.expression <- log1p(connectome$recept.expression)
  connectome$edgeweight <- connectome$ligand.expression * connectome$recept.expression
  connectome <- connectome[connectome$edgeweight > 0, ]

  source_types <- data.frame(source_type = seu@meta.data[, cell.type.calls], source = colnames(seu))
  target_types <- data.frame(receiver_type = seu@meta.data[, cell.type.calls], receiver = colnames(seu))
  connectome <- merge(connectome, target_types, by = "receiver", all.x = TRUE, all.y = FALSE)
  connectome <- merge(connectome, source_types, by = "source", all.x = TRUE, all.y = FALSE)
  connectome
}

assignInNamespace("GenerateInteractome", patched_GenerateInteractome, ns = "scriabin")
# library(scriabin) copies exported bindings into the attached "package:scriabin"
# search-path environment at attach time; assignInNamespace() alone only updates
# the namespace itself, so an unqualified GenerateInteractome() call after
# library() would still resolve to the stale copy unless we also update this.
if ("package:scriabin" %in% search()) {
  pkg_env <- as.environment("package:scriabin")
  unlockBinding("GenerateInteractome", pkg_env)
  assign("GenerateInteractome", patched_GenerateInteractome, envir = pkg_env)
  lockBinding("GenerateInteractome", pkg_env)
}
message("Patched scriabin::GenerateInteractome in-memory (slot->layer, species-indexed lr_resources.rds)")
