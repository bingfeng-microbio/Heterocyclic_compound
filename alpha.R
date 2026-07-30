rm(list=ls())#clear Global Environment
library(ggplot2)
library(ggpubr)
library(ggsignif)
library(ggprism)
library(vegan)
library(picante)
library(dplyr)
library(RColorBrewer)


df <- read.delim("genus.txt",header = TRUE,check.names = FALSE)
rownames(df) <- df[,1]
df <- df[,-1]

Shannon <- diversity(df, index = "shannon", MARGIN = 2, base = exp(1))
Simpson <- diversity(df, index = "simpson", MARGIN = 2, base = exp(1))
Richness <- specnumber(df, MARGIN = 2)  # sobs

index <- as.data.frame(cbind(Shannon, Simpson, Richness))

tdf <- ceiling(as.data.frame(t(df)))
obs_chao_ace <- t(estimateR(tdf))
obs_chao_ace <- obs_chao_ace[rownames(index),]
index$Chao <- obs_chao_ace[, 2]
index$Ace  <- obs_chao_ace[, 4]
index$obs  <- obs_chao_ace[, 1]

index$Pielou <- Shannon / log(Richness, 2)
index$Goods_coverage <- 1 - colSums(df == 1) / colSums(df)


write.table(
  cbind(sample = rownames(index), index),
  'diversity.txt',
  row.names = FALSE,
  sep = '\t',
  quote = FALSE
)


