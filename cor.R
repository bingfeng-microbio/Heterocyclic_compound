wd<- "/Users/bingfengchen/Desktop"
setwd(wd)
library(psych)

data=read.table("cor.txt", sep = "\t",head=T, row.names=1)
occor = corr.test(data,use="pairwise",method="spearman",adjust="fdr",alpha=0.05)
occor.r = occor$r
occor.p = occor$p
occor.r[occor.p>0.05|abs(occor.r)<0.7] = 0
occor.r[occor.p>0.05] = 0
write.csv(occor.r,file="cor_result.csv")
