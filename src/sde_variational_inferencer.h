#include <RcppArmadillo.h>
#include <vector>
#include <string>
#include "kernel.h"
#include "lbfgsb_cpp/problem.h"
#include "lbfgsb_cpp/l_bfgs_b.h"


class sde_variational_inferencer : public problem<arma::vec> {
public:
    sde_variational_inferencer(kernel& driftKernel, kernel& diffKernel, double v);

    sde_variational_inferencer(kernel& driftKernel, kernel& diffKernel, double v,
                               l_bfgs_b<arma::vec>& laplaceSolver,
                               l_bfgs_b<arma::vec>& lowerBoundSolver);

    ~sde_variational_inferencer() = default;

    bool get_verbose() const;

    void set_verbose(bool verbose);

    int get_max_iterations() const;

    void set_max_iterations(int maxIt);

    double get_rel_tolerance() const;

    void set_rel_tolerance(double relTolerance);

    Rcpp::List do_inference(const arma::mat& timeSeries, double samplingPeriod,
                            arma::mat& inducingPoints, int targetIndex = 0);

    double operator()(const arma::vec& x);

    double get_lower_bound() const;

private:
    // the time series
    double mSamplingPeriod;
    arma::mat mHeadX;
    // derivative of the time series
    arma::mat mDX;
    // mTarget holds the concrete dimension of mDX for which we are trying to fit
    // the dynamical terms.
    arma::vec mTarget;
    // Parameters controlling the algorithm
    int mMaxIt = 20;
    bool mVerbose = true;
    double mRelativeTolerance = 1e-5;
    // kernels
    kernel& mDriftKernel;
    kernel& mDiffKernel;
    // solvers
    l_bfgs_b<arma::vec> mLaplaceSolver;
    l_bfgs_b<arma::vec> mLowerBoundSolver;
    // results of the inference
    std::vector<double> mLikelihoodLowerBounds;
    arma::vec mInducingDriftMean;
    arma::mat mInducingDriftCov;
    arma::vec mInducingDiffMean;
    arma::mat mInducingDiffCov;
    double mV;
    arma::mat mInducingPoints;
    // auxiliar variables
    int mNoPseudoInputs;
    int mNoDriftHyperparameters;
    int mNoDiffHyperparameters;
    // The matrices of the augmented model (i.e., any x and the inducing inputs)
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

    // private methods
    double get_lower_bound(const arma::vec& dxTarget, double samplingPeriod,
                           const arma::vec& driftMean, const arma::mat& driftCov,
                           const arma::mat& A, const arma::vec& Qii,
                           const arma::mat& kmmInv, double v,
                           const arma::vec& diffMean, const arma::mat& diffCov,
                           const arma::mat& B, const arma::vec& Hii,
                           const arma::mat& jmmInv) const;

    void update_distributions();

    void calculate_model_matrices();

    static void calculate_kernel_matrices(const arma::mat& headX,
                                          const kernel& ker, const arma::mat& inducingPoints,
                                          arma::mat& kmm, arma::mat& kmmInv,
                                          arma::mat& knm, arma::mat& A, arma::vec& Qii);

    arma::vec calculate_E_vector() const;

    arma::colvec calculate_E_vector(double v,
                                    const arma::vec& diffMean,
                                    const arma::mat& diffCov,
                                    const arma::mat& B,
                                    const arma::vec& Hii) const;

    arma::vec calculate_ksi_vector() const;

    arma::colvec calculate_ksi_vector(const arma::vec& dxTarget, double samplingPeriod,
                                      const arma::vec& driftMean, const arma::mat& driftCov,
                                      const arma::mat& A, const arma::vec& Qii) const;

    arma::vec zip_hyperparameters() const;

    void unzip_hyperparameters(const arma::vec& hyperparams);

    void distributions_update_message(int nIt);

    void hyperparameters_optimization_message(int nIt);

    // A one-line print of arma::vec (instead of the column like << arma::vec
    void print_vector(const arma::vec& x, const std::string& preface = "");

    class laplace_objective_function : public problem<arma::vec> {
    public:
        laplace_objective_function(const arma::vec& cvector,
                                   const sde_variational_inferencer& inferencer);

        double operator()(const arma::vec& x);

    private:
        const arma::vec& mCVector;
        const sde_variational_inferencer& mInferencer;
    };

};
