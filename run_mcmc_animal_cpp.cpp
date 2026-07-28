// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// Helper: Multivariate Normal Draw (assumes this is already in your cpp file)
arma::vec mvrnorm_cpp(arma::vec mu, arma::mat Sigma) {
    int n = Sigma.n_cols;
    arma::mat L = arma::chol(Sigma, "lower");
    arma::vec z = arma::randn(n);
    return mu + L * z;
}

// [[Rcpp::export]]
arma::mat run_mcmc_animal_cpp(int iterations, arma::vec Y, arma::mat X, arma::mat Z,
                              arma::vec beta_init, arma::vec a_init, 
                              double sigma_e2_init, double sigma_a2_init,
                              arma::mat Sigma_prior_inv, arma::vec mu_prior,
                              double shape_prior, double scale_prior,
                              arma::mat A_inv) { // NEW: Inverse Relationship Matrix
    
    int N = Y.n_elem;
    int P = X.n_cols;
    int q = a_init.n_elem; // Number of animals in pedigree
    arma::mat samples(iterations, P + 2);
    
    arma::vec beta = beta_init;
    arma::vec a = a_init;
    double sigma_e2 = sigma_e2_init;
    double sigma_a2 = sigma_a2_init;
    
    arma::mat XtX = X.t() * X;
    arma::mat ZtZ = Z.t() * Z;
    
    for(int iter = 0; iter < iterations; iter++) {
        
        // 1. Update Fixed Effects (beta)
        arma::vec Za = Z * a;
        arma::mat V_beta = arma::inv(XtX / sigma_e2 + Sigma_prior_inv);
        arma::vec mean_beta = V_beta * (X.t() * (Y - Za) / sigma_e2 + Sigma_prior_inv * mu_prior);
        beta = mvrnorm_cpp(mean_beta, V_beta);
        
        // 2. Update Breeding Values (a) - Block Update with A_inv
        // Precision matrix for a includes the pedigree structure A_inv
        arma::mat V_a = arma::inv(ZtZ / sigma_e2 + A_inv / sigma_a2);
        arma::vec mean_a = V_a * (Z.t() * (Y - X * beta) / sigma_e2);
        a = mvrnorm_cpp(mean_a, V_a);
        
        // 3. Update Variances
        // Residual Variance
        arma::vec resid = Y - X * beta - Z * a;
        double ss_e = arma::dot(resid, resid);
        sigma_e2 = 1.0 / R::rgamma(shape_prior + N / 2.0, 1.0 / (scale_prior + ss_e / 2.0)); 
        
        // Additive Genetic Variance (Weighted by A_inv)
        double ss_a = arma::as_scalar(a.t() * A_inv * a);
        sigma_a2 = 1.0 / R::rgamma(shape_prior + q / 2.0, 1.0 / (scale_prior + ss_a / 2.0));
        
        // 4. Store parameters
        for(int p = 0; p < P; p++) samples(iter, p) = beta(p);
        samples(iter, P) = sigma_a2;
        samples(iter, P + 1) = sigma_e2;
    }
    return samples;
}