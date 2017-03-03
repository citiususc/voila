#include <RcppArmadillo.h>
#include "kernel.h"


class sde_variational_inferencer {
public:
  sde_variational_inferencer(const arma::mat& timeSeries, double samplingPeriod);
  ~sde_variational_inferencer() = default;
  // TODO: getters and setters for the algorithm parameters
  void do_inference(int targetIndex, arma::mat& xm, kernel& fKernel,
                    kernel& sKernel, double v);
  double get_lower_bound(const kernel& fKernel,
                         const kernel& sKernel,
                         const double v) const;
private:
  // the time series
  arma::mat mX;
  // all rows of mX except the last one... TODO:: highly redundant
  arma::mat mHeadX;
  // derivative of the time series
  arma::mat mDX;
  int mDXSize;
  // Parameters controlling the algorithm
  double mSamplingPeriod;
  int mMaxIt = 20;
  int mHpIt = 5;
  double mRelativeTolerance = 1e-5;
  // results of the inference
  arma::vec mPosteriorFMean;
  arma::mat mPosteriorFCov;
  arma::vec mPosteriorSMean;
  arma::mat mPosteriorSCov;
  // auxiliar variables
  int mNoPseudoInputs;
  int mNoFHyperparameters;
  int mNoSHyperparameters;
  arma::mat mKmm;
  arma::mat mKmmInv;
  arma::mat mKnm;
  arma::mat mA;
  arma::vec mQii;
  arma::mat mJmm;
  arma::mat mJmmInv;
  arma::mat mJnm;
  arma::mat mB;
  arma::vec mHii;
  // methods
  arma::vec calculate_E_vector(double v) const;
  arma::vec calculate_ksi_vector() const;
  void calculate_model_matrices(const kernel& fKer, const kernel& sKer,
                                const arma::mat& xm);

  void calculate_kernel_matrices(const kernel& ker,const arma::mat& xm,
                                 arma::mat& kmm, arma::mat& kmmInv,
                                 arma::mat& knm, arma::mat& A, arma::vec& Qii);
};
