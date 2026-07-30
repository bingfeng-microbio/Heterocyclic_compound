wd<- "/Users/bingfengchen/Desktop"
setwd(wd)
library(ggplot2)
library(vegan)

#OTU
otu <- read.delim('genus.txt', row.names = 1, sep = '\t', stringsAsFactors = FALSE, check.names = FALSE)
otu <- data.frame(t(otu))
group <- read.delim('group.txt', sep = '\t', stringsAsFactors = FALSE)
distance <- vegdist(otu, method = 'bray')
pcoa <- cmdscale(distance, k = (nrow(otu) - 1), eig = TRUE)
pcoa_eig <- (pcoa$eig)[1:2] / sum(pcoa$eig)


sample_site <- data.frame({pcoa$point})[1:2]
sample_site$names <- rownames(sample_site)
names(sample_site)[1:2] <- c('PCoA1', 'PCoA2')
sample_site <- merge(sample_site, group,by ='names', all.x = TRUE)
write.table(sample_site, file = "sample_site.tsv",sep = "\t",row.names = T,col.names = NA,quote = F)

pcoa_plot <- ggplot(sample_site, aes(PCoA1, PCoA2, group = group)) +
  theme(panel.grid = element_blank(), panel.background = element_rect(color = 'black', fill = 'transparent'), legend.key = element_rect(fill = 'transparent')) +   geom_vline(xintercept = 0, color = 'gray', linewidth = 0.3) + 
  geom_hline(yintercept = 0, color = 'gray', linewidth = 0.3) + 
  geom_point(aes(color =group, shape =group), size = 8, alpha = 1) + 
  scale_shape_manual(values = c(19,19)) +
  scale_color_manual(values = c("#74879E", "#C28484","#C7D6DE","#929292","#EDE7E2","#D4D3C4")) + #可在这里修改点的颜色
  labs(x = paste('PCoA axis1: ', round(100 * pcoa_eig[1], 2), '%'), y = paste('PCoA axis2: ', round(100 * pcoa_eig[2], 2), '%')) 
pcoa_plot <- pcoa_plot + stat_ellipse(aes(group = group), geom = "polygon", type = "t", level = 0.95, linetype = 1, color = "black", fill = NA)

pcoa_plot

#adonis
adonis_result_otu <- adonis2(otu~group, group, permutations = 999, distance = 'bray')  #这边方法与上面pcoa一样，用的BC

adonis_result_otu
df1<-data.frame(
  x=adonis_result_otu$class.vec,
  y=adonis_result_otu$dis.rank
)
write.csv(df1,file="df1.csv")
