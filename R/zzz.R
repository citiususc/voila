loadModule("KERNELS", TRUE)

#' @useDynLib sgpsde
#' @export gp_kernel exp_kernel rq_kernel exp_const_kernel sum_exp_kernels
#' @importFrom yuima simulate setModel setSampling get.zoo.data
NULL
