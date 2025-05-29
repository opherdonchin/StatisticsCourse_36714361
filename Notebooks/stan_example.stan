data {
  int<lower=0> N; 
  vector[N] y;
  real mu_mu;
  real<lower=0> sd_mu;
  real<lower=0> sd_sd; 
}

parameters {
  real mu;
  real<lower=0> sd;
}

model {
  // Priors
  mu ~ normal(mu_mu, sd_mu);
  sigma ~ normal(0, sd_sd); 

  // Likelihood
  y ~ normal(mu, sd);
}