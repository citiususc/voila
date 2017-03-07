#include <RcppArmadillo.h>
#include <vector>
#include "kernel.h"
#include "lbfgsb_cpp/problem.h"
#include "lbfgsb_cpp/l_bfgs_b.h"


class sde_variational_inferencer : public problem<arma::vec> {
public:
  sde_variational_inferencer(const arma::mat& timeSeries, double samplingPeriod);
  ~sde_variational_inferencer() = default;
  // TODO: getters and setters for the algorithm parameters
  void set_verbose_level(int verboseLevel);
  void set_max_iterations(int maxIt);
// TODO :change
  Rcpp::List do_inference(int targetIndex, arma::mat& xm, kernel& fKernel,
                    kernel& sKernel, double v);
  double operator()(const arma::vec& x);
  double get_lower_bound(double v) const;
private:
  // the time series
  arma::mat mX;
  // all rows of mX except the last one... TODO:: this is highly redundant
  arma::mat mHeadX;
  // derivative of the time series
  arma::mat mDX;
  int mDXSize;
  // Parameters controlling the algorithm
  double mSamplingPeriod;
  int mMaxIt = 20;
  int mHpIt = 5;
  bool mVerbose = true;
  double mRelativeTolerance = 1e-5;
  // results of the inference
  std::vector<double> mLikelihoodLowerBounds;
  arma::vec mPosteriorFMean;
  arma::mat mPosteriorFCov;
  arma::vec mPosteriorSMean;
  arma::mat mPosteriorSCov;
  // auxiliar variables
  l_bfgs_b<arma::vec> mSolver;
  int mVerboseHp = 0;
  kernel mFKernel;
  kernel mSKernel;
  arma::vec mTarget;
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

  static double get_lower_bound(const arma::vec& dxTarget, double samplingPeriod,
                         const arma::vec& fMean, const arma::mat& fCov,
                         const arma::mat& A, const arma::vec& Qii,
                         const arma::mat& kmmInv, double v,
                         const arma::vec& sMean, const arma::mat& sCov,
                         const arma::mat& B, const arma::vec& Hii,
                         const arma::mat& jmmInv);

  arma::vec calculate_E_vector(double v) const;

  static arma::colvec calculate_E_vector(double v,
                                  const arma::vec& sMean,
                                  const arma::mat& sCov,
                                  const arma::mat& B,
                                  const arma::vec& Hii);

  arma::vec calculate_ksi_vector() const;

  static arma::colvec calculate_ksi_vector(const arma::vec& dxTarget, double samplingPeriod,
                                    const arma::vec& fMean, const arma::mat& fCov,
                                    const arma::mat& A, const arma::vec& Qii);

  void update_distributions(double v);

  void calculate_model_matrices(const arma::mat& xm);

  static void calculate_kernel_matrices(const arma::mat& headX,
                                 const kernel& ker,const arma::mat& xm,
                                 arma::mat& kmm, arma::mat& kmmInv,
                                 arma::mat& knm, arma::mat& A, arma::vec& Qii);

  class laplace_objective_function : public problem<arma::vec> {
  public:
    laplace_objective_function(int m,
                               const arma::vec& cVector,  double v,
                               const sde_variational_inferencer& inferencer
                               )
      : problem(m), mInferencer(inferencer), mCVector(cVector), mV(v) {
    }

    double operator() (const arma::vec& x) {
      arma::colvec meanMinusV(x.n_elem);
      std::transform(x.begin(), x.end(), meanMinusV.begin(),
                     bind2nd(std::plus<double>(), -mV));
      arma::vec tmp = mInferencer.mB * meanMinusV;
      return -0.5 * (
          -1.0 / mInferencer.mSamplingPeriod * arma::dot(mCVector, arma::exp(-tmp)) -
            arma::accu(tmp) - arma::dot(meanMinusV, mInferencer.mJmmInv * meanMinusV));
    }

  private:
    arma::vec mCVector;
    double mV;
    const sde_variational_inferencer& mInferencer;
  };


  void distributions_update_message(int nIt);

  void hyperparameters_optimization_message(int nIt);
};
