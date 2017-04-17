#' @importFrom Rcpp loadModule
#' @import Rcpp methods
loadModule("VOILA_KERNELS", TRUE)

#' Gaussian process' kernels
#'
#' Available kernels for creating gaussian processes
#'
#' @details
#' \itemize{
#' \item \emph{gp_kernel} Abstract class representing a generic GP kernel. Final users
#' should not use it.
#' \item \emph{exp_kernel} An squared-exponential kernel with fixed amplitude.
#' \item \emph{rq_kernel} A rational-quadratic kernel with fixed amplitude.
#' \item \emph{exp_const_kernel} A sum of a squared-exponential kernel and a constant
#' value. The sum of the constant value and the amplitude of the exponential
#' kernel are tied so that they always sum the same value.
#' \item \emph{sum_exp_kernels} A sum of two different squared-exponential kernels:
#' The sum of the amplitudes of the exponential
#' kernels are tied so that they always sum the same value.
#' \item \emph{clamped_exp_lin_kernel} A sum of a squared-exponential kernel and a
#' linear kernel. The sum of the amplitudes of the two terms are tied so
#' that they always sum the same value. Note that this kernel is non-stationary.
#' }
#' @name gp_kernels
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