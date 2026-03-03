#!/usr/bin/env Rscript
pkg_install <- c("bit64", "psych", "ggplot2")

print("Installing R packages")
update.packages(repos = "http://cran.us.r-project.org")
pkg_installed <- installed.packages()[,"Package"]

for(pkg in pkg_install) {
    if( !pkg %in% pkg_installed) {
        install.packages(pkg, repos = "http://cran.us.r-project.org", dependencies=TRUE)
    }
}
