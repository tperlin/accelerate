#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
if (length(args)<1) {
  stop("At least one argument must be supplied\n", call.=FALSE)
}
in_filename = args[1]
data <- data.frame()
if(file.exists(in_filename)) {
	data <- read.csv(in_filename)
	names(data) <- NULL
	sd <- vector()
	if(nrow(data) > 0 && length(data) > 0) {
		values <- data.frame()
		sd <- c(sd(data[,1], na.rm=TRUE))
	} else {
		stop(paste("Empty file", in_filename, "\n"), call.=FALSE)
	}
	cat(paste(sd, "\n"))
} else{
  stop(paste("File", in_filename, "does not exist\n"), call.=FALSE)
}
