library(tidyr)
library(ggplot2)

setwd("~/bin/git/genpara/alignment scripts")
sane = read.csv('sanity_output.tsv', header = TRUE, sep = '\t')
# sane = read.csv('genpara.maxent.nobias.tsv.csv', header = FALSE, sep = ' ')

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
    ggtitle(title) +
    geom_point() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}
graphrec <- function(dataset, title){
  ggplot(data=dataset, aes(x=reorder(metric, -rec),y=rec)) +
    ggtitle(title) +
    geom_point() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}

pdfpath = "testpdf.pdf"
pdf(file = pdfpath)

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

dev.off()
