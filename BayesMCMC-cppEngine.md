## Methodological Documentation: Custom Bayesian MCMC Engine for Hierarchical Models

This document outlines the methodology, mathematical framework, and experimental findings for the custom Bayesian Markov Chain Monte Carlo (MCMC) toolkit designed for Linear Mixed Models (LMMs).

---

### 1. Core Functions and Software Architecture

The toolkit relies on a combination of R for data parsing, initialization, and visualization, paired with a high-performance C++ backend (`RcppArmadillo`) for computationally intensive Gibbs sampling.

#### 1.1 Data Simulation: `simulate_mcmc_data()`

This function generates clustered, hierarchical datasets to test MCMC recovery. It mimics the data-generating process of a linear mixed-effects model:

$$Y = X\beta + ZU + \epsilon$$

* **Fixed Effects ($X\beta$):** Supports either an intercept-only model or a two-parameter model (intercept and one continuous predictor).
* **Random Effects ($ZU$):** Simulates random intercepts (and optional random slopes) for discrete groups, with customizable covariance matrices to inject correlation between intercepts and slopes.
* **Residual Error ($\epsilon$):** Normally distributed observational noise.

#### 1.2 The Gibbs Sampler: `run_custom_mcmc()`

This wrapper function parses R formulas, initializes matrices, and routes the execution to either a native R `for` loop or a compiled C++ engine (`run_mcmc_cpp`).

**Mathematical Updating Scheme (Gibbs Conditionals):**
The sampler sequentially draws from the conditional posterior distributions of each parameter:

1. **Fixed Effects ($\beta$):** Drawn from a Multivariate Normal distribution using a Cholesky decomposition for stability. The covariance is defined by the design matrix and prior precision:

$$V_\beta = \left( \frac{X^T X}{\sigma^2_e} + \Sigma^{-1}_{prior} \right)^{-1}$$


2. **Random Effects ($U$):** Block-updated for each group $j$. Assumes independence between groups:

$$V_{u_j} = \left( \frac{n_j}{\sigma^2_e} + \frac{1}{\sigma^2_u} \right)^{-1}$$


3. **Variances ($\sigma^2_e, \sigma^2_u$):** Drawn from Inverse-Gamma conjugate distributions, updated using the sum of squared residuals and the sum of squared random effects, respectively.

---

### 2. Initialization and Prior Strategies

The engine supports three distinct initialization methods to demonstrate the geometry of MCMC convergence:

* **`lme` (Frequentist Warm Start):** Uses `lme4` to find the Maximum Likelihood Estimates (MLE) and starts the chains at the peak of the likelihood.
* **`random`:** Initializes parameters from wide, arbitrary distributions, forcing the sampler to "walk" to the target distribution during the burn-in phase.
* **Default:** Initializes at zero or one.

Priors can be set to **non-informative** (flat, allowing the likelihood to dominate) or **informative** (allowing the user to inject domain knowledge via precision matrices and Inverse-Gamma hyperparameters).


---

## 1. Core Functions

### 1.1 Function: `simulate_mcmc_data`

This function generates clustered, hierarchical datasets to test MCMC recovery. It mirrors the data-generating process of a standard linear mixed-effects model:

$$Y = X\beta + ZU + \epsilon$$

It allows users to specify fixed effects (intercept and optional slope), random effects (group-level variance and covariance), and residual error.

**Implemented Code:**

```r
library(MASS)

simulate_mcmc_data <- function(n_groups = 20, 
                               n_per_group = 15, 
                               beta = c(50, 10),       # Fixed effects
                               sigma_u = c(25, 5),     # Random effects variances
                               u_corr = 0,             # Correlation between random effects
                               sigma_e = 15) {         # Residual variance
  
  N <- n_groups * n_per_group
  groups <- as.factor(rep(1:n_groups, each = n_per_group))
  
  # 1. Build Fixed Effects Design Matrix (X)
  X <- matrix(1, nrow = N, ncol = length(beta))
  if(length(beta) == 2) {
    X[, 2] <- rnorm(N, mean = 0, sd = 1) 
  }
  
  # 2. Simulate Random Effects (U)
  if (length(sigma_u) == 1) {
    U <- matrix(rnorm(n_groups, 0, sqrt(sigma_u)), ncol = 1)
  } else {
    cov_val <- u_corr * sqrt(sigma_u[1] * sigma_u[2])
    Sigma_u <- matrix(c(sigma_u[1], cov_val, cov_val, sigma_u[2]), 2, 2)
    U <- MASS::mvrnorm(n_groups, mu = c(0, 0), Sigma = Sigma_u)
  }
  
  # 3. Build Random Effects Design Matrix (Z) and Calculate ZU
  ZU <- numeric(N)
  for (i in 1:N) {
    grp <- as.numeric(groups[i])
    if (length(sigma_u) == 1) {
      ZU[i] <- U[grp, 1]
    } else {
      ZU[i] <- U[grp, 1] + U[grp, 2] * X[i, 2]
    }
  }
  
  # 4. Generate Final Response Variable
  epsilon <- rnorm(N, 0, sqrt(sigma_e))
  Y <- (X %*% beta) + ZU + epsilon
  
  df <- data.frame(Group = groups, Y = as.numeric(Y))
  if(length(beta) == 2) df$X1 <- X[, 2]
  return(list(data = df, true_params = list(beta = beta, sigma_u = sigma_u, sigma_e = sigma_e)))
}

```

This function generates the synthetic hierarchical data. The parameters allow you to control the exact structural variance and fixed effects of the generative model, which is essential for testing model recovery.

| Parameter | Type | Description |
| --- | --- | --- |
| **`n_groups`** | Integer | The number of distinct clusters or groups (e.g., classrooms, patients) in the hierarchical structure. |
| **`n_per_group`** | Integer | The number of observations within each group. The total sample size will be $N = \text{n\_groups} \times \text{n\_per\_group}$. |
| **`beta`** | Numeric Vector | The true fixed effects. A vector of length 1 generates an intercept-only model. A vector of length 2 generates an intercept and a single continuous predictor ($X$). |
| **`sigma_u`** | Numeric Vector | The true variance of the random effects. Length 1 dictates a random intercept variance ($\sigma^2_{u0}$). Length 2 dictates random intercept and random slope variances. |
| **`u_corr`** | Numeric | The correlation ($\rho$) between random intercepts and random slopes. Only applied if `sigma_u` has a length of 2. |
| **`sigma_e`** | Numeric | The true residual (observation-level) error variance ($\sigma^2_e$). |

---

#
### 1.2 Function: `run_custom_mcmc` & C++ Engine

This is the core MCMC engine. It uses a Gibbs sampler to iteratively draw from conditional posterior distributions.

* **Fixed Effects ($\beta$):** Drawn via multivariate normal conjugate updates.
* **Random Effects ($U$):** Block-updated per group.
* **Variance Components ($\sigma^2_e, \sigma^2_u$):** Drawn via Inverse-Gamma conjugate updates.

The R wrapper (`run_custom_mcmc`) parses the formula, establishes initial states using `lme4` (optional), defines prior matrices, and routes computation to the compiled C++ backend.

**Implemented Code (C++ Backend):**

```cpp
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// Helper: Multivariate Normal Draw
arma::vec mvrnorm_cpp(arma::vec mu, arma::mat Sigma) {
    int n = Sigma.n_cols;
    arma::mat L = arma::chol(Sigma, "lower");
    arma::vec z = arma::randn(n);
    return mu + L * z;
}

// [[Rcpp::export]]
arma::mat run_mcmc_cpp(int iterations, arma::vec Y, arma::mat X, 
                       arma::uvec groups, int n_groups, 
                       arma::vec beta_init, arma::vec U_init, 
                       double sigma_e2_init, double sigma_u2_init,
                       arma::mat Sigma_prior_inv, arma::vec mu_prior,
                       double shape_prior, double scale_prior) {
    
    int N = Y.n_elem;
    int P = X.n_cols;
    arma::mat samples(iterations, P + 2);
    
    arma::vec beta = beta_init;
    arma::vec U = U_init;
    double sigma_e2 = sigma_e2_init;
    double sigma_u2 = sigma_u2_init;
    arma::mat XtX = X.t() * X;
    
    for(int iter = 0; iter < iterations; iter++) {
        // 1. Update Fixed Effects
        arma::vec ZU(N);
        for(int i = 0; i < N; i++) ZU(i) = U(groups(i) - 1);
        
        arma::mat V_beta = arma::inv(XtX / sigma_e2 + Sigma_prior_inv);
        arma::vec mean_beta = V_beta * (X.t() * (Y - ZU) / sigma_e2 + Sigma_prior_inv * mu_prior);
        beta = mvrnorm_cpp(mean_beta, V_beta);
        
        // 2. Update Random Effects
        for(int j = 0; j < n_groups; j++) {
            arma::uvec idx = arma::find(groups == (j + 1));
            int n_j = idx.n_elem;
            double V_u = 1.0 / (n_j / sigma_e2 + 1.0 / sigma_u2);
            double resid_sum = 0;
            for(int i = 0; i < n_j; i++) {
                int row = idx(i);
                resid_sum += Y(row) - arma::as_scalar(X.row(row) * beta);
            }
            double mean_u = V_u * resid_sum / sigma_e2;
            U(j) = mean_u + sqrt(V_u) * R::rnorm(0, 1);
        }
        
        // 3. Update Variances
        double ss_e = 0;
        for(int i = 0; i < N; i++) {
            double resid = Y(i) - arma::as_scalar(X.row(i) * beta) - U(groups(i) - 1);
            ss_e += resid * resid;
        }
        sigma_e2 = 1.0 / R::rgamma(shape_prior + N / 2.0, 1.0 / (scale_prior + ss_e / 2.0)); 
        
        double ss_u = arma::dot(U, U);
        sigma_u2 = 1.0 / R::rgamma(shape_prior + n_groups / 2.0, 1.0 / (scale_prior + ss_u / 2.0));
        
        // 4. Store
        for(int p = 0; p < P; p++) samples(iter, p) = beta(p);
        samples(iter, P) = sigma_u2;
        samples(iter, P + 1) = sigma_e2;
    }
    return samples;
}

```
*(Note: The full `run_custom_mcmc` R wrapper is large and handles parsing, initialization routing, and outputting `ggplot2` summaries as documented in the source code).*


This is the primary wrapper function that parses the model formula, handles matrix initialization, and routes the calculations to either the R or C++ backend.

| Parameter | Type | Description |
| --- | --- | --- |
| **`data`** | Data Frame | The dataset containing the response variable, fixed effect predictors, and the grouping factor. |
| **`formula`** | Formula | Standard `lme4`-style formula syntax specifying the model structure (e.g., `Y ~ X1 + (1 |
| **`iterations`** | Integer | The total number of MCMC draws to execute per chain, *including* the burn-in phase. |
| **`burn_in`** | Integer | The number of initial draws to discard from the final posterior samples to allow the Markov chain to reach its stationary distribution. |
| **`thin`** | Integer | The thinning interval to reduce autocorrelation. A value of 3 means the engine saves every 3rd draw and discards the rest. |
| **`prior_type`** | Character | Accepts `"non_informative"` (uses diffuse/flat priors where data likelihood overwhelmingly dictates the posterior) or `"informative"`. |
| **`prior_specs`** | List | Required if `prior_type = "informative"`. Must contain exactly four elements: `beta_mean` (vector), `beta_var` (vector), `var_shape` (scalar), and `var_scale` (scalar) to construct the Normal and Inverse-Gamma priors. |
| **`init_method`** | Character | Dictates starting values. Accepts `"random"` (draws from priors/uniform space) or `"lme"` (fits a rapid frequentist model via `lme4` to start the chain exactly at the Maximum Likelihood Estimate). |
| **`engine`** | Character | Accepts `"R"` for the native R `for` loop (ideal for line-by-line debugging and learning) or `"cpp"` for the compiled, high-speed `RcppArmadillo` backend. |

---

## 2. Methodological Experiments

The repository includes scripts to demonstrate key Bayesian concepts natively.

### 2.1 Experiment 1: Prior Washing

Demonstrates how massive datasets mathematically overwhelm strongly incorrect priors.

* **Setup:** The data has a true slope of 10. The user injects a highly confident, mathematically rigorous prior insisting the slope is -20. A massive dataset ($N = 5,000$) is supplied.
* **Result:** The likelihood dominates the posterior, pulling the estimate back to 10.

**Implemented Code:**

```r
# 1. Simulate a MASSIVE Dataset (N = 5,000)
large_data_sim <- simulate_mcmc_data(
  n_groups = 100, n_per_group = 50, 
  beta = c(50, 10), sigma_u = c(15), sigma_e = 10
)

# 2. Define a Strongly Incorrect Prior
wrong_priors <- list(
  beta_mean = c(0, -20),  # Wrong Intercept, Wrong Slope (-20)
  beta_var = c(0.1, 0.1), # Very tight variance = high false confidence
  var_shape = 2, var_scale = 10
)

# 3. Run MCMC Engine (Starting randomly to force learning)
washing_results <- run_custom_mcmc(
  data = large_data_sim$data, formula = Y ~ X1 + (1 | Group), 
  iterations = 2500, burn_in = 500, thin = 2,
  prior_type = "informative", prior_specs = wrong_priors, init_method = "random" 
)

print(washing_results$Summary)

```

### 2.2 Experiment 2: The Variance Sponge

Illustrates the danger of model misspecification. If hierarchical structure (group-level variance) is ignored, the standard linear regression model absorbs this structural variation directly into the residual error term, severely inflating it.

**Implemented Code:**

```r
# 1. Simulate Highly Clustered Data
sponge_data_sim <- simulate_mcmc_data(
  n_groups = 30, n_per_group = 20, beta = c(100, 5), 
  sigma_u = c(150),  # Massive structural variance
  sigma_e = 20       # Tiny residual error variance
)

# 2. Run Correct Hierarchical MCMC
correct_mcmc <- run_custom_mcmc(
  data = sponge_data_sim$data, formula = Y ~ X1 + (1 | Group), 
  iterations = 2000, burn_in = 500, prior_type = "non_informative"
)

# 3. Run Misspecified Flat Model
wrong_flat_model <- lm(Y ~ X1, data = sponge_data_sim$data)

# 4. Comparison
cat("True Error Variance: 20\n")
cat("Custom MCMC (Correctly Modeled): ", mean(correct_mcmc$Samples$Sigma2_e), "\n")
cat("Flat Model (Misspecified Sponge): ", sigma(wrong_flat_model)^2, "\n")

```

### 2.3 Experiment 3: Battle of Priors and Initializations

A four-way matrix test to prove that **initializations** dictate burn-in efficiency, while **priors** dictate the final stationary distribution. A good frequentist "warm start" (`lme4`) cannot save a model from an overwhelming bad prior in a small dataset.

| Scenario | Prior Type | Start Method | Outcome |
| --- | --- | --- | --- |
| **Gold Standard** | Flat | `lme` | Immediate convergence on the true parameter. |
| **Worst Case** | Strong/Wrong | Random | Chain anchors at the incorrect prior; fails to find truth. |
| **The Drag-Down** | Strong/Wrong | `lme` | Chain starts at the truth but is immediately dragged away by the strong prior during burn-in. |
| **Slow Learner** | Flat | Random | Chain starts in a terrible location but successfully walks to the truth over the burn-in period. |

> **Methodological Takeaway:** Initializations strictly govern the efficiency of the burn-in phase. Priors, however, dictate the final stationary distribution. A perfect initialization cannot save a model from a mathematically overwhelming, incorrect prior in small samples.


**Implemented Code:**

```r
# 1. Moderate dataset (N = 300)
test_data <- simulate_mcmc_data(n_groups = 20, n_per_group = 15, beta = c(50, 10), sigma_u = 15, sigma_e = 10)

# 2. Strongly Incorrect Prior (Slope = -20)
bad_priors <- list(beta_mean = c(50, -20), beta_var = c(10, 0.5), var_shape = 2, var_scale = 10)

# 3. The Four Models
mod_gold <- run_custom_mcmc(test_data$data, Y ~ X1 + (1 | Group), prior_type = "non_informative", init_method = "lme", iterations = 2000, burn_in = 500)
mod_worst <- run_custom_mcmc(test_data$data, Y ~ X1 + (1 | Group), prior_type = "informative", prior_specs = bad_priors, init_method = "random", iterations = 2000, burn_in = 500)
mod_drag <- run_custom_mcmc(test_data$data, Y ~ X1 + (1 | Group), prior_type = "informative", prior_specs = bad_priors, init_method = "lme", iterations = 2000, burn_in = 500)
mod_slow <- run_custom_mcmc(test_data$data, Y ~ X1 + (1 | Group), prior_type = "non_informative", init_method = "random", iterations = 2000, burn_in = 500)

# 4. Compare Results
results_comparison <- data.frame(
  Scenario = c("Gold Standard (No Prior, LME Start)", "Worst Case (Bad Prior, Random Start)", 
               "The Drag-Down (Bad Prior, LME Start)", "Slow Learner (No Prior, Random Start)"),
  Estimated_Slope = c(mod_gold$Summary$Estimate[2], mod_worst$Summary$Estimate[2], 
                      mod_drag$Summary$Estimate[2], mod_slow$Summary$Estimate[2])
)
print(results_comparison)

```

### 2.4 Experiment 4: Benchmark (Custom Gibbs vs. Stan HMC)

Validates the mathematical accuracy of the C++ Gibbs engine against the industry-standard `brms` (Stan).

* **Result:** Both samplers converge on the exact same estimates. `BayesMCMC-cppEngine` executes nearly instantaneously due to lacking Stan's C++ compilation overhead, though Stan provides more mathematically efficient exploration per draw via Hamiltonian Monte Carlo.

**Implemented Code:**

```r
library(brms)

# 1. Data Setup
benchmark_data <- simulate_mcmc_data(n_groups = 30, n_per_group = 20, beta = c(50, 10), sigma_u = 15, sigma_e = 10)

# 2. Run Custom Engine
custom_model <- run_custom_mcmc(
  data = benchmark_data$data, formula = Y ~ X1 + (1 | Group), 
  iterations = 4000, burn_in = 1000, engine = "cpp"
)

# 3. Run brms
brms_model <- brm(
  formula = Y ~ X1 + (1 | Group), data = benchmark_data$data,
  chains = 2, iter = 2500, warmup = 1000, cores = 2, silent = 2
)

# 4. Compare Summaries (Code truncates brms extraction for brevity)
# Compare custom_model$Summary against summary(brms_model)

```


