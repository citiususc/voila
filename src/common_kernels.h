#ifndef GPSDE_EXP_KERNEL_H
#define GPSDE_EXP_KERNEL_H

#include "kernel.h"

inline void check_length_scales_dims(int inputDimension, const arma::vec& lengthScales) {
  if (inputDimension != lengthScales.n_elem) {
    throw std::invalid_argument("The size of the lengthScales vector should match \
                                  the input dimension of the kernel");
  }
}

class exponential_kernel: public kernel {
public:
  exponential_kernel(int inputDimension, double Amplitude,
                     const arma::vec& lengthScales,
                     double epsilon = 0.0);
private:
  static kernel::kernel_expression generate_exponential_kernel(double amplitude);
};

class sum_exponential_kernels: public kernel {
public:
  sum_exponential_kernels(int inputDimension, double completeAmplitude,
                          double A1,
                          const arma::vec& lengthScales1,
                          const arma::vec& lengthScales2,
                          double epsilon = 0.0);
private:
  static kernel::kernel_expression generate_sum_exponential_kernels(double amplitude);

};

class rational_quadratic_kernel: public kernel {
public:
  rational_quadratic_kernel(int inputDimension,
                            double amplitude,
                            double alpha, double lengthScale,
                            double epsilon = 0.0);
private:
  static kernel::kernel_expression generate_rational_quadratic_kernel(double amplitude);
};


class exponential_constant_kernel: public kernel {
public:
  exponential_constant_kernel(int inputDimension,
                              double completeAmplitude,
                              double expAmplitude,
                              const arma::vec& lengthScales,
                              double epsilon = 0.0);
private:
  static kernel::kernel_expression generate_expression(double completeAmplitude);
};




#endif
