install.packages('tidyverse')
install.packages('rstatix')
install.packages('ellipsis')
install.packages('vctrs')
install.packages('ggpubr')
install.packages('phytools')
install.packages('remotes')
install.packages('devtools')
install.packages('ggraph')
install.packages('ggdendro')
install.packages('ape')
install.packages('cowplot')


if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("ggtree")

library(dplyr)
library(tidyverse)
library(car)
library(rstatix)
library(ggplot2)
library(RColorBrewer)
library(ggpubr)
library(ggtree)
library(phytools)
library(devt)
library(ggraph)
library(ggdendro)
library(ape)
library(cowplot)

set.seed(1024)

cos_sim$org <- factor(cos_sim$org, levels=c('E. coli', 'P. aeruginosa', 'Streptomyces sps.', 'S. cerevisiae', 'M. musculus'))

p <- ggplot(cos_sim, aes(x=score, y=org, fill=org)) +
      geom_boxplot() +
      scale_fill_brewer(palette = 'Dark2') +
      labs(x='Cosine Similarity', y='Orgasnisms') +
      guides(fill=guide_legend(title = "Organisms")) +
      theme_minimal()

p

plot(umap_antismash$Dim1, umap_antismash$Dim2)

data_summary <- cos_sim %>%
                  group_by(org) %>%
                  get_summary_stats(score, type='mean_sd')

data_summary

data_outliers <- cos_sim %>%
                  group_by(org) %>%
                  identify_outliers(score)

data_outliers # S. cerevisiae has extreme outliers

# ANOVA Model
aov <- lm(score ~ org, data = cos_sim)

# Assumptions

# 1. Normality
qq <- qplot(sample = score, data = cos_sim, color = factor(org), facets = .~org) +
        scale_color_brewer(palette = 'Dark2') + 
        labs(x='theoretical residues', y='sample residues', color = 'Organisms')
qq

shapiro.test(residuals(aov)) # p-value = 5.47e-06, can't assume independence

normality_test <- cos_sim %>%
                    group_by(org) %>%
                    shapiro_test(score)
normality_test

# Homogeneity of Variance
plot(aov, 1)

homo_var <- cos_sim %>% levene_test(score ~ org) # p-value > 0.05
homo_var

res_aov <- cos_sim %>% anova_test(score ~ org) # significant differences between groups
res_aov


# Post-Hoc tests

pwc <- cos_sim %>% tukey_hsd(score ~ org)
pwc

# Welch One wat ANOVA test
res_aov2 <- cos_sim %>% welch_anova_test(score ~ org)
res_aov2

file <- file.choose()

remove.packages('dplyr')
devtools::install_version("dplyr",version="1.0.5")
require(dplyr)

tanimoto <- subset_mibig_tanimono[1:239]

name <- subset_mibig_tanimono['label']
bio <- subset_mibig_tanimono['biosyn_class']

hc <- hclust(dist(tanimoto), method='complete')
# hc <- as.dendrogram(hc)

hc$labels <- bio$biosyn_class
class(hc)
my_tree <- as.phylo(hc)

write.tree(phy=my_tree, file='complete_tanimoto_tree.newick')

my_tree <- as.phylo(ddata)

tmp <- umap_antismash
tmp$label <- as.factor(tmp$label)


e1 <- ggplot(tmp, aes(x=Dim1, y=Dim2, color=label, group=label)) +
        geom_point(aes(color=label), show.legend = FALSE) +
        scale_colour_brewer(name="BGC Class", palette = "Dark2", labels=c('PKS','NRPS', 'Terpene Synthases','RiPPs')) +
        theme_minimal()
e1



l <- get_legend(e1)
l

tmp <- embedding_chebyshev_chebyshev_45_0_2_0_3
tmp$biosyn_class <- as.factor(tmp$biosyn_class)

e2 <- ggplot(tmp, aes(x=Dim1, y=Dim2, color=biosyn_class, group=biosyn_class)) +
  geom_point(aes(color=biosyn_class), show.legend = FALSE) +
  scale_colour_brewer(palette = "Dark2") +
  theme_minimal()

e2

tmp <- embedding_chebyshev_euclidean_30_0_2_0_3
tmp$biosyn_class <- as.factor(tmp$biosyn_class)

e3 <- ggplot(tmp, aes(x=Dim1, y=Dim2, color=biosyn_class, group=biosyn_class)) +
  geom_point(aes(color=biosyn_class), show.legend = FALSE) +
  scale_colour_brewer(palette = "Dark2") +
  theme_minimal()

e3


tmp <- embedding_euclidean_chebyshev_15_0_1_0_3
tmp$biosyn_class <- as.factor(tmp$biosyn_class)


e4 <- ggplot(tmp, aes(x=Dim1, y=Dim2, color=biosyn_class, group=biosyn_class)) +
        geom_point(aes(color=biosyn_class), show.legend = FALSE) +
        scale_colour_brewer(palette = "Dark2") +
        theme_minimal()

e4

ecombine <- plot_grid(e1, e2, e3, e4, labels = c('A', 'B', 'C', 'D'), label_size = 12)
ecombine <- plot_grid(ecombine, l, ncol = 1, rel_heights = c(1, .1))
ecombine
