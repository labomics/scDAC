library(Seurat)
library(SeuratDisk)
library(Signac)
library(future)
library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)
library(dplyr)
library(ggplot2)
library(Matrix)
library(purrr)
library(stringr)
library(GenomicRanges)
library(RcppTOML)
library(ymlthis)
library(argparse)
set.seed(1234)
plan("multicore", workers = 64)
options(future.globals.maxSize = 100 * 1024^3) # for 100 Gb RAM
options(future.seed = T)
pj <- file.path #file.path（a, b）把a与b用“/”连接起来形成一个路径。




prt <- function(...) {
    cat(paste0(...))
}


plt_size <- function(w, h) {
     options(repr.plot.width = w, repr.plot.height = h)
}


mkdir <- function(directory, remove_old = F) {
    if (remove_old) {
        if (dir.exists(directory)) {
             prt("Removing directory ", directory, "\n")
             unlink(directory, recursive = T)
        }
    }
    if (!dir.exists(directory)) {
        dir.create(directory, recursive = T)
    }
}


mkdirs <- function(directories, remove_old = F) {
    for (directory in directories) {
        mkdir(directory, remove_old = remove_old)
    }
}


gen_atac <- function(frag_path, min_cells = 5) {
    # call peaks using MACS2
    system(paste0("tabix -f -p bed ", frag_path))
    frags <- CreateFragmentObject(frag_path)
    peaks <- CallPeaks(frags)
    peaks@seqnames
    # remove peaks on non-autosomes and in genomic blacklist regions
    peaks <- keepStandardChromosomes(peaks, pruning.mode = "coarse")
    peaks <- peaks[!(peaks@seqnames %in% c("chrX", "chrY"))]
    peaks <- subsetByOverlaps(x = peaks, ranges = blacklist_hg38_unified, invert = TRUE)
    # quantify counts in each peak
    atac_counts <- FeatureMatrix(
        fragments = frags,
        features = peaks
    )
    # # add in the atac-seq data, only use peaks in standard chromosomes
    # grange <- StringToGRanges(rownames(atac_counts))
    # grange_use <- seqnames(grange) %in% standardChromosomes(grange)
    # atac_counts <- atac_counts[as.vector(grange_use), ]
    # get gene annotations for hg38
    ### ATAC analysis add gene annotation information
    annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
    seqlevelsStyle(annotation) <- "UCSC"
    genome(annotation) <- "hg38"
    # create atac assay and add it to the object
    atac_assay <- CreateChromatinAssay(
        counts = atac_counts,
        min.cells = min_cells,
        genome = 'hg38',
        fragments = frags,
        annotation = annotation
    )
    atac <- CreateSeuratObject(
        counts = atac_assay,
        assay = 'atac',
    )
    atac <- NucleosomeSignal(atac)
    atac <- TSSEnrichment(atac)
    return(atac)
}


gen_rna <- function(rna_counts, min_cells = 3) {
    rna <- CreateSeuratObject(
        counts = rna_counts,
        min.cells = min_cells,
        assay = "rna"
    ) #创建seurat文件
    rna[["percent.mt"]] <- PercentageFeatureSet(rna, pattern = "^MT-") #其中的.mt列是用这个特征基因数除以基因总数
    return(rna)
}


gen_adt <- function(adt_counts) {
    # rename features
    feat <- unlist(map(rownames(adt_counts), tolower))
    feat <- unlist(map(feat, gsub, pattern = "-|_|\\(|\\)|/", replacement = "."))
    feat <- unlist(map(feat, gsub, pattern = "^cd3$", replacement = "cd3.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd4$", replacement = "cd4.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd11b$", replacement = "cd11b.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd26$", replacement = "cd26.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd38$", replacement = "cd38.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.$", replacement = "cd56.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd56.ncam.recombinant$", replacement = "cd56.2"))
    feat <- unlist(map(feat, gsub, pattern = "^cd57.recombinant$", replacement = "cd57"))
    feat <- unlist(map(feat, gsub, pattern = "^cd90.thy1.$", replacement = "cd90"))
    feat <- unlist(map(feat, gsub, pattern = "^cd112.nectin.2.$", replacement = "cd112"))
    feat <- unlist(map(feat, gsub, pattern = "^cd117.c.kit.$", replacement = "cd117"))
    feat <- unlist(map(feat, gsub, pattern = "^cd138.1.syndecan.1.$", replacement = "cd138.1"))
    feat <- unlist(map(feat, gsub, pattern = "^cd155.pvr.$", replacement = "cd155"))
    feat <- unlist(map(feat, gsub, pattern = "^cd269.bcma.$", replacement = "cd269"))
    feat <- unlist(map(feat, gsub, pattern = "^clec2$", replacement = "clec1b"))
    feat <- unlist(map(feat, gsub, pattern = "^cadherin11$", replacement = "cadherin"))
    feat <- unlist(map(feat, gsub, pattern = "^folate.receptor$", replacement = "folate"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.1$", replacement = "notch1"))
    feat <- unlist(map(feat, gsub, pattern = "^notch.2$", replacement = "notch3"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.a.b$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.2$", replacement = "tcrab"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.g.d$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.1$", replacement = "tcrgd"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va7.2$", replacement = "tcr.v.7.2"))
    feat <- unlist(map(feat, gsub, pattern = "^tcr.va24.ja18$", replacement = "tcr.v.24.j.18"))
    feat <- unlist(map(feat, gsub, pattern = "^vegfr.3$", replacement = "vegfr3"))
    # feat <- unlist(map(feat, gsub, pattern = "^igg1.k.isotype.control$", replacement = "rat.igg1k.isotypectrl"))
    rownames(adt_counts) <- feat
    # remove features
    adt_counts <- adt_counts[-grep("igg", rownames(adt_counts)), ]
    # create adt object
    adt <- CreateSeuratObject(
      counts = adt_counts,
      assay = "adt"
    )
    return(adt)
}


preprocess <- function(output_dir, atac = NULL, rna = NULL, adt = NULL) {
    # preprocess and save data
    if (!is.null(atac)) {
        atac <- RunTFIDF(atac) %>%
                FindTopFeatures(min.cutoff = "q0")
        SaveH5Seurat(atac, pj(output_dir, "atac.h5seurat"), overwrite = TRUE)
    } #### We exclude the first dimension as this is typically correlated with sequencing depth

    if (!is.null(rna)) {
        rna <- NormalizeData(rna) %>%
               FindVariableFeatures(nfeatures = 4000) %>%
               ScaleData()
        SaveH5Seurat(rna, pj(output_dir, "rna.h5seurat"), overwrite = TRUE)
    } ### Perform standard analysis of each modality independently RNA analysis

    if (!is.null(adt)) {
        VariableFeatures(adt) <- rownames(adt)
        adt <- NormalizeData(adt, normalization.method = "CLR", margin = 2) %>%
               ScaleData()
        SaveH5Seurat(adt, pj(output_dir, "adt.h5seurat"), overwrite = TRUE)
    }
}