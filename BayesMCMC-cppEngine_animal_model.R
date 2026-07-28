# ==========================================
# 1. Load Dependencies
# ==========================================
library(Rcpp)
library(RcppArmadillo)
library(nadiv)
library(MASS)

# ==========================================
# 2. Embed and Compile the C++ Engine
# ==========================================
cpp_code <- '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// Helper function for Multivariate Normal draws
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
                              double shape_a, double scale_a, 
                              double shape_e, double scale_e, 
                              arma::mat A_inv) { 
    
    int N = Y.n_elem;
    int P = X.n_cols;
    int q = a_init.n_elem; 
    
    // MODIFIED: Expanded matrix to store q breeding values
    arma::mat samples(iterations, P + 2 + q);
    
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
        
        // 2. Update Breeding Values (a)
        arma::mat V_a = arma::inv(ZtZ / sigma_e2 + A_inv / sigma_a2);
        arma::vec mean_a = V_a * (Z.t() * (Y - X * beta) / sigma_e2);
        a = mvrnorm_cpp(mean_a, V_a);
        
        // 3. Update Variances 
        arma::vec resid = Y - X * beta - Z * a;
        double ss_e = arma::dot(resid, resid);
        sigma_e2 = 1.0 / R::rgamma(shape_e + N / 2.0, 1.0 / (scale_e + ss_e / 2.0)); 
        
        double ss_a = arma::as_scalar(a.t() * A_inv * a);
        sigma_a2 = 1.0 / R::rgamma(shape_a + q / 2.0, 1.0 / (scale_a + ss_a / 2.0));
        
        // 4. Store
        for(int p = 0; p < P; p++) samples(iter, p) = beta(p);
        samples(iter, P) = sigma_a2;
        samples(iter, P + 1) = sigma_e2;
        
        // MODIFIED: Store the breeding values (EBVs)
        for(int i = 0; i < q; i++) samples(iter, P + 2 + i) = a(i); 
    }
    return samples;
}
'

# Compile the C++ code on the fly
Rcpp::sourceCpp(code = cpp_code)

# ==========================================
# 3. The Smart R Wrapper
# ==========================================
fit_animal_model <- function(Y, X, Z, A_inv, iterations = 6000, 
                             expected_h2 = NULL, degree_of_belief = 5) {
  
  # Base Initializations
  V_P <- var(Y)
  sigma_e2_init <- V_P * 0.5
  sigma_a2_init <- V_P * 0.5
  beta_init <- rep(0, ncol(X))
  a_init <- rep(0, ncol(Z))
  
  # Default Priors
  Sigma_prior_inv <- diag(1e-6, ncol(X)) 
  mu_prior <- rep(0, ncol(X))
  
  # Prior Routing Logic
  if (is.null(expected_h2)) {
    message("Fitting with DEFAULT diffuse priors (shape = 0.001, scale = 0.001)")
    shape_a <- 0.001; scale_a <- 0.001
    shape_e <- 0.001; scale_e <- 0.001
  } else {
    if(expected_h2 <= 0 || expected_h2 >= 1) stop("expected_h2 must be strictly between 0 and 1.")
    message(sprintf("Fitting with INFORMED priors (Target h^2 = %.2f, Belief = %d)", 
                    expected_h2, degree_of_belief))
    
    expected_sigma_a2 <- V_P * expected_h2
    expected_sigma_e2 <- V_P * (1 - expected_h2)
    
    shape_a <- degree_of_belief; scale_a <- expected_sigma_a2 * (shape_a - 1)
    shape_e <- degree_of_belief; scale_e <- expected_sigma_e2 * (shape_e - 1)
  }
  
  # Execute Engine
  samples <- run_mcmc_animal_cpp(
    iterations = iterations, 
    Y = Y, 
    X = X, 
    Z = Z, 
    beta_init = beta_init, 
    a_init = a_init, 
    sigma_e2_init = sigma_e2_init, 
    sigma_a2_init = sigma_a2_init,
    Sigma_prior_inv = Sigma_prior_inv, 
    mu_prior = mu_prior,
    shape_a = shape_a, scale_a = scale_a,
    shape_e = shape_e, scale_e = scale_e,
    A_inv = A_inv
  )
  
  # MODIFIED: Label columns including the new breeding values
  P <- ncol(X)
  q <- ncol(Z)
  colnames(samples) <- c(paste0("Beta", 0:(P-1)), "Sigma2_a", "Sigma2_e", paste0("a_", 1:q))
  
  return(samples)
}

# ==========================================
# 4. Simulation Function
# ==========================================
simulate_animal_data <- function(pedigree, beta, sigma_a2, sigma_e2, obs_per_animal = 1) {
  
  A_sparse <- nadiv::makeA(pedigree)
  A_matrix <- as.matrix(A_sparse)
  n_animals <- nrow(pedigree)
  
  true_a <- MASS::mvrnorm(1, mu = rep(0, n_animals), Sigma = A_matrix * sigma_a2)
  
  sim_data <- data.frame(
    animal_id = rep(pedigree$id, each = obs_per_animal),
    X1 = rnorm(n_animals * obs_per_animal, mean = 0, sd = 1)
  )
  
  X <- model.matrix(~ 1 + X1, data = sim_data)
  Z <- model.matrix(~ 0 + as.factor(animal_id), data = sim_data)
  
  sim_data$Y <- as.numeric(X %*% beta) + 
    as.numeric(Z %*% true_a) + 
    rnorm(nrow(sim_data), mean = 0, sd = sqrt(sigma_e2))
  
  A_inv_matrix <- as.matrix(nadiv::makeAinv(pedigree)$Ainv)
  
  return(list(
    data = sim_data,
    X = X,
    Z = Z,
    true_a = true_a,
    A_inv = A_inv_matrix
  ))
}

# ==========================================
# A. Build a multi-generational pedigree
# ==========================================
sires <- 1:10
dams <- 11:60
offspring <- 61:260

pedigree <- data.frame(
  id = 1:260,
  sire = c(rep(NA, 60), rep(sires, each = 20)),
  dam  = c(rep(NA, 60), rep(dams, each = 4))
)

# ==========================================
# B. Simulate Data
# ==========================================
true_beta <- c(Intercept = 100, Slope = 15)
true_sigma_a2 <- 40
true_sigma_e2 <- 60

set.seed(42)
sim <- simulate_animal_data(
  pedigree = pedigree, 
  beta = true_beta, 
  sigma_a2 = true_sigma_a2, 
  sigma_e2 = true_sigma_e2, 
  obs_per_animal = 2 
)

# ==========================================
# C. Run the Model (Choose Diffuse or Informed)
# ==========================================
# Diffuse
samples_diffuse <- fit_animal_model(
  Y = sim$data$Y, X = sim$X, Z = sim$Z, A_inv = sim$A_inv, iterations = 6000
)

# Or Informed h^2
samples_informed <- fit_animal_model(
  Y = sim$data$Y, X = sim$X, Z = sim$Z, A_inv = sim$A_inv, iterations = 6000,
  expected_h2 = 0.30, degree_of_belief = 10 
)

# Using diffuse for validation below
samples <- samples_diffuse 

# ==========================================
# D. Validation and Diagnostics
# ==========================================
burn_in <- 1000

# MODIFIED: Subset to structural parameters only (exclude EBVs) for the table
struct_cols <- 1:4 
posteriors_struct <- samples[(burn_in + 1):6000, struct_cols]

results_table <- data.frame(
  Parameter = c("Beta0 (Intercept)", "Beta1 (Slope)", "Sigma^2_a", "Sigma^2_e"),
  True_Value = c(true_beta[1], true_beta[2], true_sigma_a2, true_sigma_e2),
  Posterior_Mean = colMeans(posteriors_struct),
  CrI_Lower = apply(posteriors_struct, 2, quantile, probs = 0.025),
  CrI_Upper = apply(posteriors_struct, 2, quantile, probs = 0.975)
)
print(results_table)

plot_animal_diagnostics <- function(samples, burn_in = 1000, true_values = NULL) {
  n_params <- ncol(samples)
  iterations <- nrow(samples)
  
  P <- n_params - 2
  param_names <- character(n_params)
  for(i in 1:P) param_names[i] <- paste0("Beta", i-1)
  param_names[n_params - 1] <- "Sigma^2_a (Genetic)"
  param_names[n_params]     <- "Sigma^2_e (Residual)"
  
  old_par <- par(mfrow = c(n_params, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par)) 
  
  for (i in 1:n_params) {
    plot(1:iterations, samples[, i], type = "l", col = "steelblue",
         main = paste("Trace:", param_names[i]), xlab = "Iteration", ylab = "Value")
    abline(v = burn_in, col = "darkred", lwd = 2, lty = 2)
    if (!is.null(true_values)) abline(h = true_values[i], col = "forestgreen", lwd = 2)
    
    post_burn_in_samples <- samples[(burn_in + 1):iterations, i]
    dens <- density(post_burn_in_samples)
    plot(dens, main = paste("Density:", param_names[i]),
         xlab = "Value", ylab = "Density", col = "darkorange", lwd = 2)
    polygon(dens, col = adjustcolor("darkorange", alpha.f = 0.3), border = NA)
    if (!is.null(true_values)) abline(v = true_values[i], col = "forestgreen", lwd = 2)
  }
  mtext("MCMC Diagnostics", outer = TRUE, cex = 1.2, font = 2)
}

# MODIFIED: Pass only the 4 structural columns to the plot to prevent plotting 260 EBVs
true_params <- c(100, 15, 40, 60)
plot_animal_diagnostics(samples[, struct_cols], burn_in = 1000, true_values = true_params)

# ==========================================
# E.  Calculate Heritability and EBVs
# ==========================================
post_samples <- samples[-(1:burn_in), ]

# --- 1. Posterior Heritability (h^2) ---
sigma2_a_chain <- post_samples[, "Sigma2_a"]
sigma2_e_chain <- post_samples[, "Sigma2_e"]
h2_chain <- sigma2_a_chain / (sigma2_a_chain + sigma2_e_chain)

posterior_mean_h2 <- mean(h2_chain)
credible_interval_h2 <- quantile(h2_chain, probs = c(0.025, 0.975))

cat(sprintf("\nPosterior Heritability: %.3f (95%% CrI: %.3f - %.3f)\n", 
            posterior_mean_h2, credible_interval_h2[1], credible_interval_h2[2]))

# --- 2. Estimated Breeding Values (EBVs) ---
# Extract just the 'a_' columns
a_samples <- post_samples[, grepl("^a_", colnames(post_samples))]

animal_evaluations <- data.frame(
  Animal_ID = 1:ncol(a_samples),
  True_a = sim$true_a,
  EBV = colMeans(a_samples),
  Lower_95 = apply(a_samples, 2, quantile, probs = 0.025),
  Upper_95 = apply(a_samples, 2, quantile, probs = 0.975)
)

cat("\nTop 5 Animals by EBV:\n")
print(head(animal_evaluations[order(-animal_evaluations$EBV), ]))

# # ==========================================
# 1. Load Dependencies
# ==========================================
library(Rcpp)
library(RcppArmadillo)
library(nadiv)
library(MASS)

# ==========================================
# 2. Embed and Compile the C++ Engine
# ==========================================
cpp_code <- '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// Helper function for Multivariate Normal draws
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
                              double shape_a, double scale_a, 
                              double shape_e, double scale_e, 
                              arma::mat A_inv) { 
    
    int N = Y.n_elem;
    int P = X.n_cols;
    int q = a_init.n_elem; 
    
    // MODIFIED: Expanded matrix to store q breeding values
    arma::mat samples(iterations, P + 2 + q);
    
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
        
        // 2. Update Breeding Values (a)
        arma::mat V_a = arma::inv(ZtZ / sigma_e2 + A_inv / sigma_a2);
        arma::vec mean_a = V_a * (Z.t() * (Y - X * beta) / sigma_e2);
        a = mvrnorm_cpp(mean_a, V_a);
        
        // 3. Update Variances 
        arma::vec resid = Y - X * beta - Z * a;
        double ss_e = arma::dot(resid, resid);
        sigma_e2 = 1.0 / R::rgamma(shape_e + N / 2.0, 1.0 / (scale_e + ss_e / 2.0)); 
        
        double ss_a = arma::as_scalar(a.t() * A_inv * a);
        sigma_a2 = 1.0 / R::rgamma(shape_a + q / 2.0, 1.0 / (scale_a + ss_a / 2.0));
        
        // 4. Store
        for(int p = 0; p < P; p++) samples(iter, p) = beta(p);
        samples(iter, P) = sigma_a2;
        samples(iter, P + 1) = sigma_e2;
        
        // MODIFIED: Store the breeding values (EBVs)
        for(int i = 0; i < q; i++) samples(iter, P + 2 + i) = a(i); 
    }
    return samples;
}
'

# Compile the C++ code on the fly
Rcpp::sourceCpp(code = cpp_code)

# ==========================================
# 3. The Smart R Wrapper
# ==========================================
fit_animal_model <- function(Y, X, Z, A_inv, iterations = 6000, 
                             expected_h2 = NULL, degree_of_belief = 5) {
  
  # Base Initializations
  V_P <- var(Y)
  sigma_e2_init <- V_P * 0.5
  sigma_a2_init <- V_P * 0.5
  beta_init <- rep(0, ncol(X))
  a_init <- rep(0, ncol(Z))
  
  # Default Priors
  Sigma_prior_inv <- diag(1e-6, ncol(X)) 
  mu_prior <- rep(0, ncol(X))
  
  # Prior Routing Logic
  if (is.null(expected_h2)) {
    message("Fitting with DEFAULT diffuse priors (shape = 0.001, scale = 0.001)")
    shape_a <- 0.001; scale_a <- 0.001
    shape_e <- 0.001; scale_e <- 0.001
  } else {
    if(expected_h2 <= 0 || expected_h2 >= 1) stop("expected_h2 must be strictly between 0 and 1.")
    message(sprintf("Fitting with INFORMED priors (Target h^2 = %.2f, Belief = %d)", 
                    expected_h2, degree_of_belief))
    
    expected_sigma_a2 <- V_P * expected_h2
    expected_sigma_e2 <- V_P * (1 - expected_h2)
    
    shape_a <- degree_of_belief; scale_a <- expected_sigma_a2 * (shape_a - 1)
    shape_e <- degree_of_belief; scale_e <- expected_sigma_e2 * (shape_e - 1)
  }
  
  # Execute Engine
  samples <- run_mcmc_animal_cpp(
    iterations = iterations, 
    Y = Y, 
    X = X, 
    Z = Z, 
    beta_init = beta_init, 
    a_init = a_init, 
    sigma_e2_init = sigma_e2_init, 
    sigma_a2_init = sigma_a2_init,
    Sigma_prior_inv = Sigma_prior_inv, 
    mu_prior = mu_prior,
    shape_a = shape_a, scale_a = scale_a,
    shape_e = shape_e, scale_e = scale_e,
    A_inv = A_inv
  )
  
  # MODIFIED: Label columns including the new breeding values
  P <- ncol(X)
  q <- ncol(Z)
  colnames(samples) <- c(paste0("Beta", 0:(P-1)), "Sigma2_a", "Sigma2_e", paste0("a_", 1:q))
  
  return(samples)
}

# ==========================================
# 4. Simulation Function
# ==========================================
simulate_animal_data <- function(pedigree, beta, sigma_a2, sigma_e2, obs_per_animal = 1) {
  
  A_sparse <- nadiv::makeA(pedigree)
  A_matrix <- as.matrix(A_sparse)
  n_animals <- nrow(pedigree)
  
  true_a <- MASS::mvrnorm(1, mu = rep(0, n_animals), Sigma = A_matrix * sigma_a2)
  
  sim_data <- data.frame(
    animal_id = rep(pedigree$id, each = obs_per_animal),
    X1 = rnorm(n_animals * obs_per_animal, mean = 0, sd = 1)
  )
  
  X <- model.matrix(~ 1 + X1, data = sim_data)
  Z <- model.matrix(~ 0 + as.factor(animal_id), data = sim_data)
  
  sim_data$Y <- as.numeric(X %*% beta) + 
    as.numeric(Z %*% true_a) + 
    rnorm(nrow(sim_data), mean = 0, sd = sqrt(sigma_e2))
  
  A_inv_matrix <- as.matrix(nadiv::makeAinv(pedigree)$Ainv)
  
  return(list(
    data = sim_data,
    X = X,
    Z = Z,
    true_a = true_a,
    A_inv = A_inv_matrix
  ))
}

# ==========================================
# A. Build a multi-generational pedigree
# ==========================================
sires <- 1:10
dams <- 11:60
offspring <- 61:260

pedigree <- data.frame(
  id = 1:260,
  sire = c(rep(NA, 60), rep(sires, each = 20)),
  dam  = c(rep(NA, 60), rep(dams, each = 4))
)

# ==========================================
# B. Simulate Data
# ==========================================
true_beta <- c(Intercept = 100, Slope = 15)
true_sigma_a2 <- 40
true_sigma_e2 <- 60

set.seed(42)
sim <- simulate_animal_data(
  pedigree = pedigree, 
  beta = true_beta, 
  sigma_a2 = true_sigma_a2, 
  sigma_e2 = true_sigma_e2, 
  obs_per_animal = 2 
)

# ==========================================
# C. Run the Model (Choose Diffuse or Informed)
# ==========================================
# Diffuse
samples_diffuse <- fit_animal_model(
  Y = sim$data$Y, X = sim$X, Z = sim$Z, A_inv = sim$A_inv, iterations = 6000
)

# Or Informed h^2
samples_informed <- fit_animal_model(
  Y = sim$data$Y, X = sim$X, Z = sim$Z, A_inv = sim$A_inv, iterations = 6000,
  expected_h2 = 0.30, degree_of_belief = 10 
)

# Using diffuse for validation below
samples <- samples_diffuse 

# ==========================================
# D. Validation and Diagnostics
# ==========================================
burn_in <- 1000

# MODIFIED: Subset to structural parameters only (exclude EBVs) for the table
struct_cols <- 1:4 
posteriors_struct <- samples[(burn_in + 1):6000, struct_cols]

results_table <- data.frame(
  Parameter = c("Beta0 (Intercept)", "Beta1 (Slope)", "Sigma^2_a", "Sigma^2_e"),
  True_Value = c(true_beta[1], true_beta[2], true_sigma_a2, true_sigma_e2),
  Posterior_Mean = colMeans(posteriors_struct),
  CrI_Lower = apply(posteriors_struct, 2, quantile, probs = 0.025),
  CrI_Upper = apply(posteriors_struct, 2, quantile, probs = 0.975)
)
print(results_table)

plot_animal_diagnostics <- function(samples, burn_in = 1000, true_values = NULL) {
  n_params <- ncol(samples)
  iterations <- nrow(samples)
  
  P <- n_params - 2
  param_names <- character(n_params)
  for(i in 1:P) param_names[i] <- paste0("Beta", i-1)
  param_names[n_params - 1] <- "Sigma^2_a (Genetic)"
  param_names[n_params]     <- "Sigma^2_e (Residual)"
  
  old_par <- par(mfrow = c(n_params, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old_par)) 
  
  for (i in 1:n_params) {
    plot(1:iterations, samples[, i], type = "l", col = "steelblue",
         main = paste("Trace:", param_names[i]), xlab = "Iteration", ylab = "Value")
    abline(v = burn_in, col = "darkred", lwd = 2, lty = 2)
    if (!is.null(true_values)) abline(h = true_values[i], col = "forestgreen", lwd = 2)
    
    post_burn_in_samples <- samples[(burn_in + 1):iterations, i]
    dens <- density(post_burn_in_samples)
    plot(dens, main = paste("Density:", param_names[i]),
         xlab = "Value", ylab = "Density", col = "darkorange", lwd = 2)
    polygon(dens, col = adjustcolor("darkorange", alpha.f = 0.3), border = NA)
    if (!is.null(true_values)) abline(v = true_values[i], col = "forestgreen", lwd = 2)
  }
  mtext("MCMC Diagnostics", outer = TRUE, cex = 1.2, font = 2)
}

# MODIFIED: Pass only the 4 structural columns to the plot to prevent plotting 260 EBVs
true_params <- c(100, 15, 40, 60)
plot_animal_diagnostics(samples[, struct_cols], burn_in = 1000, true_values = true_params)

# ==========================================
# E. Calculate Heritability and EBVs
# ==========================================
post_samples <- samples[-(1:burn_in), ]

# --- 1. Posterior Heritability (h^2) ---
sigma2_a_chain <- post_samples[, "Sigma2_a"]
sigma2_e_chain <- post_samples[, "Sigma2_e"]
h2_chain <- sigma2_a_chain / (sigma2_a_chain + sigma2_e_chain)

posterior_mean_h2 <- mean(h2_chain)
credible_interval_h2 <- quantile(h2_chain, probs = c(0.025, 0.975))

cat(sprintf("\nPosterior Heritability: %.3f (95%% CrI: %.3f - %.3f)\n", 
            posterior_mean_h2, credible_interval_h2[1], credible_interval_h2[2]))

# --- 2. Estimated Breeding Values (EBVs) ---
# Extract just the 'a_' columns
a_samples <- post_samples[, grepl("^a_", colnames(post_samples))]

animal_evaluations <- data.frame(
  Animal_ID = 1:ncol(a_samples),
  True_a = sim$true_a,
  EBV = colMeans(a_samples),
  Lower_95 = apply(a_samples, 2, quantile, probs = 0.025),
  Upper_95 = apply(a_samples, 2, quantile, probs = 0.975)
)

cat("\nTop 5 Animals by EBV:\n")
print(head(animal_evaluations[order(-animal_evaluations$EBV), ]))

#################
# impact of four different prior setups
# We will use the coda package to calculate ESS. The true heritability in this simulation is 0.40 ($40 / (40 + 60)$).
# Load coda for Effective Sample Size (ESS) calculations
if(!require(coda)) install.packages("coda"); library(coda)

# ==========================================
# 1. Setup Weaker Simulation (1 observation per animal)
# ==========================================
# True heritability (h^2) = 40 / 100 = 0.40
true_sigma_a2 <- 40
true_sigma_e2 <- 60

set.seed(123)
sim_weak <- simulate_animal_data(
  pedigree = pedigree, 
  beta = c(100, 15), 
  sigma_a2 = true_sigma_a2, 
  sigma_e2 = true_sigma_e2, 
  obs_per_animal = 1 # Weaker data = stronger prior influence
)

# ==========================================
# 2. Run the Four Models
# ==========================================
iterations <- 5000
burn_in <- 1000
# We use a degree of belief of 15 to make the prior moderately strong
belief <- 15 

cat("Fitting Model 1: Diffuse Prior...\n")
mod_diffuse <- fit_animal_model(Y = sim_weak$data$Y, X = sim_weak$X, Z = sim_weak$Z, 
                                A_inv = sim_weak$A_inv, iterations = iterations)

cat("Fitting Model 2: Prior h^2 = 0.10 (Wrong - Low)...\n")
mod_p10 <- fit_animal_model(Y = sim_weak$data$Y, X = sim_weak$X, Z = sim_weak$Z, 
                            A_inv = sim_weak$A_inv, iterations = iterations,
                            expected_h2 = 0.10, degree_of_belief = belief)

cat("Fitting Model 3: Prior h^2 = 0.40 (Correct)...\n")
mod_p40 <- fit_animal_model(Y = sim_weak$data$Y, X = sim_weak$X, Z = sim_weak$Z, 
                            A_inv = sim_weak$A_inv, iterations = iterations,
                            expected_h2 = 0.40, degree_of_belief = belief)

cat("Fitting Model 4: Prior h^2 = 0.80 (Wrong - High)...\n")
mod_p80 <- fit_animal_model(Y = sim_weak$data$Y, X = sim_weak$X, Z = sim_weak$Z, 
                            A_inv = sim_weak$A_inv, iterations = iterations,
                            expected_h2 = 0.80, degree_of_belief = belief)

# ==========================================
# 3. Analyze and Compare
# ==========================================
evaluate_model <- function(samples, prior_name) {
  # Discard burn-in
  post <- samples[-(1:burn_in), ]
  
  # Calculate h^2 chain
  h2_chain <- post[, "Sigma2_a"] / (post[, "Sigma2_a"] + post[, "Sigma2_e"])
  
  # Calculate metrics
  mean_h2 <- mean(h2_chain)
  cri <- quantile(h2_chain, probs = c(0.025, 0.975))
  ess <- coda::effectiveSize(h2_chain)
  
  data.frame(
    Prior_Setup = prior_name,
    Posterior_Mean = round(mean_h2, 3),
    CrI_Lower = round(cri[1], 3),
    CrI_Upper = round(cri[2], 3),
    CrI_Width = round(cri[2] - cri[1], 3),
    ESS = round(ess, 0)
  )
}

# Compile results into a single table
experiment_results <- rbind(
  evaluate_model(mod_diffuse, "Diffuse (Flat)"),
  evaluate_model(mod_p10, "Informed: h^2 = 0.10"),
  evaluate_model(mod_p40, "Informed: h^2 = 0.40 (True)"),
  evaluate_model(mod_p80, "Informed: h^2 = 0.80")
)

cat("\n======================================================\n")
cat("EXPERIMENT RESULTS (True h^2 = 0.40)\n")
cat("======================================================\n")
print(experiment_results)

# ==========================================
# 4. Visual Comparison of Posteriors
# ==========================================
# ==========================================
# 4. Visual Comparison of Posteriors (Dynamic Limits)
# ==========================================
plot_comparative_h2 <- function(mod_diffuse, mod_p10, mod_p40, mod_p80, burn_in = 1000) {
  
  # Helper function to extract h^2 chain
  get_h2 <- function(model) {
    post <- model[-(1:burn_in), ]
    return(post[, "Sigma2_a"] / (post[, "Sigma2_a"] + post[, "Sigma2_e"]))
  }
  
  # 1. Calculate densities first
  d_diffuse <- density(get_h2(mod_diffuse))
  d_p10     <- density(get_h2(mod_p10))
  d_p40     <- density(get_h2(mod_p40))
  d_p80     <- density(get_h2(mod_p80))
  
  # 2. Find the maximum y-value across all curves and add 10% headroom
  max_y <- max(c(d_diffuse$y, d_p10$y, d_p40$y, d_p80$y)) * 1.1
  
  # 3. Plot with the dynamic ylim
  plot(d_diffuse, 
       main = "Posterior h^2 Distributions by Prior", 
       xlab = "Heritability", 
       ylab = "Density",
       col = "black", lwd = 2, lty = 2, 
       xlim = c(0, 1), 
       ylim = c(0, max_y)) # Dynamic limit applied here
  
  # Add the other lines
  lines(d_p10, col = "blue", lwd = 2)
  lines(d_p40, col = "forestgreen", lwd = 2)
  lines(d_p80, col = "red", lwd = 2)
  
  # Add true value and legend
  abline(v = 0.40, col = "darkgray", lwd = 2, lty = 3)
  legend("topright", 
         legend = c("Diffuse", "Prior = 0.10", "Prior = 0.40 (True)", "Prior = 0.80", "True Value"), 
         col = c("black", "blue", "forestgreen", "red", "darkgray"), 
         lwd = 2, lty = c(2, 1, 1, 1, 3))
}

# Run the updated plotting function
plot_comparative_h2(mod_diffuse, mod_p10, mod_p40, mod_p80, burn_in = 1000)

# Benchmarking 
# ==========================================
# 5. Implementation & Benchmarking vs MCMCglmm
# ==========================================
# Install and load MCMCglmm for comparison
if(!require(MCMCglmm)) install.packages("MCMCglmm"); library(MCMCglmm)

# Ensure the simulated data is ready (uses 'sim' object from Section 4)
# True Parameters: Sigma2_a = 40, Sigma2_e = 60, h^2 = 0.40

# --- A. Setup MCMCglmm Data & Priors ---
# MCMCglmm requires specific column names and factors for the pedigree
ped_mcmc <- pedigree
names(ped_mcmc)[1] <- "animal" 
ped_mcmc$animal <- as.factor(ped_mcmc$animal)
sim$data$animal <- as.factor(sim$data$animal_id)

# Define standard uninformative Inverse-Wishart priors for MCMCglmm
prior_mcmc <- list(G = list(G1 = list(V = 1, nu = 0.002)), 
                   R = list(V = 1, nu = 0.002))

iterations <- 6000
burn_in <- 1000

# --- B. Run Timed Benchmarks ---

cat("\nRunning established package: MCMCglmm...\n")
time_mcmcglmm <- system.time({
  mod_mcmcglmm <- MCMCglmm(Y ~ 1 + X1, 
                           random = ~animal, 
                           pedigree = ped_mcmc, 
                           data = sim$data, 
                           prior = prior_mcmc, 
                           nitt = iterations, 
                           thin = 1, 
                           burnin = burn_in, 
                           verbose = FALSE)
})

cat("Running custom package: BayesMCMC-cppEngine...\n")
time_custom <- system.time({
  mod_custom <- fit_animal_model(Y = sim$data$Y, 
                                 X = sim$X, 
                                 Z = sim$Z, 
                                 A_inv = sim$A_inv, 
                                 iterations = iterations, 
                                 expected_h2 = NULL) # Null uses diffuse prior for fair comparison
})

# --- C. Extract and Compare Results ---

# Extract Custom Engine Results
post_custom <- mod_custom[-(1:burn_in), ]
custom_sigma2_a <- mean(post_custom[, "Sigma2_a"])
custom_sigma2_e <- mean(post_custom[, "Sigma2_e"])
custom_h2 <- custom_sigma2_a / (custom_sigma2_a + custom_sigma2_e)

# Extract MCMCglmm Results
mcmcglmm_sigma2_a <- mean(mod_mcmcglmm$VCV[, "animal"])
mcmcglmm_sigma2_e <- mean(mod_mcmcglmm$VCV[, "units"])
mcmcglmm_h2 <- mcmcglmm_sigma2_a / (mcmcglmm_sigma2_a + mcmcglmm_sigma2_e)

# --- D. Compile Benchmark Table ---
benchmark_results <- data.frame(
  Engine = c("True Value", "BayesMCMC-cppEngine", "MCMCglmm"),
  Time_Seconds = c(NA, round(time_custom["elapsed"], 2), round(time_mcmcglmm["elapsed"], 2)),
  Sigma2_a = c(40.00, round(custom_sigma2_a, 2), round(mcmcglmm_sigma2_a, 2)),
  Sigma2_e = c(60.00, round(custom_sigma2_e, 2), round(mcmcglmm_sigma2_e, 2)),
  Heritability = c(0.400, round(custom_h2, 3), round(mcmcglmm_h2, 3))
)

cat("\n======================================================\n")
cat("BENCHMARK RESULTS (6,000 Iterations)\n")
cat("======================================================\n")
print(benchmark_results)
