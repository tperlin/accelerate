#!/usr/bin/env Rscript
options(scipen = 999)
library(ggplot2)

args = commandArgs(trailingOnly=TRUE)
if (length(args)<2) {
  stop("At least two arguments must be supplied", call.=FALSE)
}
in_filename = args[1]
out_filename = args[2]
title = args[3]
subtitle = args[4]

if(file.exists(in_filename)){
  data <- read.csv(in_filename)
} else{
  stop(paste("File does not exist: ", in_filename, sep=""), call.=FALSE)
}

p <- ggplot(data=data, aes(x=factor(objects), y=throughput, ymin=throughput-sd, ymax=throughput+sd, fill=stm)) +
    scale_fill_brewer(palette="Paired") +
    scale_color_brewer(palette="Paired") +
    geom_bar(position=position_dodge(width=0.9), stat = "identity") + 
    geom_errorbar(width=.4,position = position_dodge(.9), colour="darkgrey") +
    labs(title="", subtitle="", caption = "", x="accounts per transaction", y="throughput (tx/s)", tag="") +
    guides(fill=guide_legend(title="")) +
    theme_minimal() +
    theme(legend.position = "bottom") ;

pdf(paste(out_filename,".bar.err.pdf",sep=""), width = 6, height = 4.5, bg = "white", colormodel = "cmyk")
print(p)
garbage <- dev.off()
