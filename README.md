# BayesMCMC-cppEngine

> A custom, high-performance Bayesian MCMC Gibbs sampler for Linear Mixed Models (LMMs). This project bridges the gap between educational methodology and computational speed by pairing an accessible R frontend with a blazing-fast C++ (`RcppArmadillo`) backend.

---

## Objective

The primary objective of **BayesMCMC-cppEngine** is to demystify Bayesian hierarchical modeling. Standard packages like `brms` or `rstanarm` act as black boxes, making it difficult for users to understand how Markov Chain Monte Carlo (MCMC) algorithms actually traverse posterior distributions. This repository provides a fully transparent, heavily documented Gibbs sampling engine that is both computationally robust for real-world application and explicitly designed as an educational tool.

---

## Methodological Details

The engine estimates parameters for Linear Mixed Models of the form:

$$Y = X\beta + ZU + \epsilon$$

It relies on a Gibbs sampling architecture, updating parameters sequentially by drawing from their conditional posterior distributions:

* **Fixed Effects (** $\beta$ **):** Drawn from a Multivariate Normal distribution. The covariance matrix is inverted efficiently using a Cholesky decomposition for numerical stability.
* **Random Effects (** $U$ **):** Block-updated for each discrete group, allowing the sampler to handle hierarchical structures cleanly.
* **Variance Components (** $\sigma^2_e, \sigma^2_u$ **):** Drawn from Inverse-Gamma conjugate distributions, utilizing the sum of squared residuals and the sum of squared random effects.

---

## Advantages of this Approach

* **Computational Speed:** By routing the computationally heavy `for` loops through `RcppArmadillo`, the MCMC chains execute nearly instantaneously (often under a second), avoiding the heavy C++ compilation overhead required by Hamiltonian Monte Carlo (HMC) engines like Stan for simple models.
* **Total Transparency:** The entire algorithm is written in readable R and C++ scripts rather than hidden in package binaries. You can modify the specific mathematical updates directly.
* **Automated Diagnostics:** Out-of-the-box integration with `ggplot2` to immediately generate traceplots, posterior density overlays, and summary statistic tables without requiring secondary diagnostic packages.

---

## What is New: Hybrid Initializations & Sandboxing

This engine introduces specific methodological features geared toward understanding MCMC behavior:

* **Frequentist "Warm Starts" (`lme` Initialization):** Unlike traditional Bayesian software that initializes chains at zero or random draws, this engine includes an `init_method = "lme"` argument. This uses the `lme4` package to rapidly calculate the Maximum Likelihood Estimates (MLE) and starts the MCMC chain exactly at the peak of the likelihood. This essentially eliminates the need for long burn-in phases and demonstrates the geometric relationship between frequentist optimization and Bayesian sampling.
* **Methodological Sandboxes:** The codebase is pre-loaded with reproducible experiments demonstrating advanced statistical concepts, including:
* **Prior Washing:** Overwhelming incorrect, high-confidence priors with massive datasets.
* **The Variance Sponge:** Demonstrating how ignoring hierarchical structure inflates residual error.
* **The Drag-Down:** Proving that an `lme` warm start cannot save a model from a mathematically overwhelming, incorrect prior during the burn-in phase.



---

## Example Implementation

The following example demonstrates how to simulate hierarchical data, define informative priors, and run the compiled C++ Gibbs sampler using a Frequentist warm start.

```r
# Load necessary libraries
library(MASS)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(Rcpp)
library(RcppArmadillo)

# Source the custom functions (assuming they are in your working directory)
# source("simulate_mcmc_data.R")
# source("run_custom_mcmc.R")

# 1. Simulate Hierarchical Data
# 30 groups, 10 observations per group. True Intercept = 45, True Slope = 12.
sim_results <- simulate_mcmc_data(
  n_groups = 30, 
  n_per_group = 10, 
  beta = c(45, 12),    
  sigma_u = c(20),     
  sigma_e = 15         
)

my_data <- sim_results$data

# 2. Define Custom Informative Priors (Optional)
# Injecting domain knowledge: We suspect intercept is ~40 and slope is ~10
my_priors <- list(
  beta_mean = c(40, 10), 
  beta_var = c(2, 2),    
  var_shape = 2, 
  var_scale = 10
)

# 3. Run the MCMC Engine using C++ and an 'lme' warm start
mcmc_results <- run_custom_mcmc(
  data = my_data, 
  formula = Y ~ X1 + (1 | Group), 
  iterations = 4000, 
  burn_in = 1000, 
  thin = 3,                       
  prior_type = "informative",     
  prior_specs = my_priors,
  init_method = "lme",            # Use lme4 to find MLE starting values
  engine = "cpp"                  # Route to high-speed RcppArmadillo backend
)

# 4. View Results
print(mcmc_results$Summary)

# 5. Render Diagnostics
print(mcmc_results$Traceplot)
print(mcmc_results$Density)

```
