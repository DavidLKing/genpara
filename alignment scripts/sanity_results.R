# srcs in corr 165 of 782
# targs in corr 169 of 782

library(tidyr)
library(ggplot2)

setwd("~/bin/git/genpara/alignment scripts")
sane = read.csv('sanity_output.tsv', header = TRUE, sep = '\t')

# EXAMPLE OF HOW TO ORDER
# test[order(test$sane.glove_src_para_sim),]
# select(sane[order(sane$glove_src_para_sim),], glove_src_para_sim)



# sane$met_prec = paste(sane$metric,sane$percent)

tens = subset(sane, percent == 10)
twenties = subset(sane, percent == 20)
thirties = subset(sane, percent == 30)
forties = subset(sane, percent == 40)
fifties = subset(sane, percent == 50)
sixty = subset(sane, percent == 60)
seventy = subset(sane, percent == 70)
eighty = subset(sane, percent == 80)
ninety = subset(sane, percent == 90)
all = subset(sane, percent == 100)
# tens = tens[order(tens$prec)]

# plot(x=tens$metric, y=tens$prec)

graphprec <- function(dataset, title){
  ggplot(data=dataset, aes(x=reorder(metric, -prec),y=prec)) +
  # ggplot(data=dataset, aes(x=metric,y=prec)) +
    ggtitle(title) +
    geom_point() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}
graphrec <- function(dataset, title){
  ggplot(data=dataset, aes(x=reorder(metric, -rec),y=rec)) +
  # ggplot(data=dataset, aes(x=metric,y=rec)) +
    ggtitle(title) +
    geom_point() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}
graphavg <- function(dataset, title){
  ggplot(data=dataset, aes(x=reorder(metric, -AveP),y=AveP)) +
  # ggplot(data=dataset, aes(x=metric,y=AveP)) +
    ggtitle(title) +
    geom_point() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}
graphmap <- function(dataset, title){
  ggplot(data=dataset, aes(x=reorder(metric, -MAP),y=MAP)) +
  # ggplot(data=dataset, aes(x=metric,y=MAP)) +
    ggtitle(title) +
    geom_point() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}

# pdfpath = "testpdf.pdf"
# pdf(file = pdfpath)

graphprec(tens, "Precision at 10%")
graphprec(twenties, "Precision at 20%")
graphprec(thirties, "Precision at 30%")
graphprec(forties, "Precision at 40%")
graphprec(fifties, "Precision at 50%")
graphprec(sixty, "Precision at 60%")
graphprec(seventy, "Precision at 70%")
graphprec(eighty, "Precision at 80%")
graphprec(ninety, "Precision at 90%")
graphprec(all, "Precision at 100%")

graphrec(tens, "Recall at 10%")
graphrec(twenties, "Recall at 20%")
graphrec(thirties, "Recall at 30%")
graphrec(forties, "Recall at 40%")
graphrec(fifties, "Recall at 50%")
graphrec(sixty, "Recall at 60%")
graphrec(seventy, "Recall at 70%")
graphrec(eighty, "Recall at 80%")
graphrec(ninety, "Recall at 90%")
graphrec(all, "Recall at 100%")

graphavg(tens, "AvgP at 10%")
graphavg(twenties, "AvgP at 20%")
graphavg(thirties, "AvgP at 30%")
graphavg(forties, "AvgP at 40%")
graphavg(fifties, "AvgP at 50%")
graphavg(sixty, "AvgP at 60%")
graphavg(seventy, "AvgP at 70%")
graphavg(eighty, "AvgP at 80%")
graphavg(ninety, "AvgP at 90%")
graphavg(all, "AvgP at 100%")

graphmap(tens, "MAP at 10%")
graphmap(twenties, "MAP at 20%")
graphmap(thirties, "MAP at 30%")
graphmap(forties, "MAP at 40%")
graphmap(fifties, "MAP at 50%")
graphmap(sixty, "MAP at 60%")
graphmap(seventy, "MAP at 70%")
graphmap(eighty, "MAP at 80%")
graphmap(ninety, "MAP at 90%")
graphmap(all, "MAP at 100%")

# dev.off()
