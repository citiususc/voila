#' @importFrom Rcpp loadModule
#' @import Rcpp methods
loadModule("VOILA_KERNELS", TRUE)

#' @useDynLib voila
#' @export gp_kernel exp_kernel rq_kernel exp_const_kernel sum_exp_kernels clamped_exp_lin_kernel
NULL


#' Dansgaard-Oeschger (DO) events time series
#'
#' A time series of delta-O-18 measurements (interpreted as temperature proxy)
#' during  the last glacial period and showing abrupt climate fluctuations
#' known as Dansgaard-Oeschger (DO) events.
#'
#' @format A \emph{ts} object
#' @source \url{ftp://ftp.ncdc.noaa.gov/pub/data/paleo/icecore/greenland/summit/ngrip/isotopes/ngrip-d18o-50yr.txt}
"do_events"