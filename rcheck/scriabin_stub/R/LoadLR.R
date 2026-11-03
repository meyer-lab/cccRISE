# Copied verbatim from scriabin R/utils.R at bce695b (LoadLR, %notin%).
#' @importFrom magrittr %>%
LoadLR <- function(species = "human", database = "OmniPath", ligands = NULL, recepts = NULL, lit_support = 7) {
  if(database=="custom") {
    message("Using custom database")
    ligands <- ligands
    recepts <- recepts
    lit.put <- data.frame(pair.name = paste(ligands,recepts,sep="_"), source_genesymbol = ligands, target_genesymbol = recepts)
  }
  else {
    if(species %notin% c("human","mouse","rat")) {
      stop("Only human, mouse, and rat supported as species")
    }
    all <- readRDS(system.file(package = "scriabin", "lr_resources.rds"))[[species]]
    if(database %notin% names(all)) {
      stop("Database must be one of: OmniPath, CellChatDB, CellPhoneDB, Ramilowski2015, Baccin2019, LRdb, Kirouac2010, ICELLNET, iTALK, EMBRACE, HPMR, Guide2Pharma, connectomeDB2020, talklr, CellTalkDB\nFor rat or mouse, only OmniPath is supported")
    }
    message(paste("Using database",database))
    pairs <- all[[database]] %>% dplyr::filter(curation_effort>=lit_support) %>%
      dplyr::select(source_genesymbol,target_genesymbol) %>% mutate_all(as.character) %>% as.data.frame()
    lit.put <- pairs %>% dplyr::mutate(pair = paste(source_genesymbol,target_genesymbol, sep = "_"))
    lit.put <- as.data.frame(lit.put[,c("pair","source_genesymbol","target_genesymbol")])
  }
  return(lit.put)
}

`%notin%` <- Negate(`%in%`)
