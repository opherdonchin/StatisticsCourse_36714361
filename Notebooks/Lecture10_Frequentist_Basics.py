# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lecture 10: Frequentist Basics
# 
# This notebook demonstrates key frequentist methods:
# - Point estimation (mean and variance)
# - Maximum likelihood estimation (MLE) vs. unbiased estimators
# - Confidence intervals and coverage
# - Frequentist linear regression
# - Comparisons with Bayesian methods where relevant
# 
# Each step is clearly marked, and we use consistent datasets throughout.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import arviz as az
import pymc as pm
import bambi as bmb

az.style.use("arviz-whitegrid")
plt.rc("figure", dpi=150)
np.random.seed(123)

# %% [markdown]
# ## 1. Mean and Variance Estimators

# %% [markdown]
# We use the chemical shifts dataset for all examples.

# %%
url = "https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/chemical_shifts_theo_exp.csv"
cs_data = pd.read_csv(url)
diff = cs_data.theo - cs_data.exp

print(f"Mean: {np.mean(diff):.2f}")
print(f"Variance: {np.var(diff, ddof=1):.2f}")

# %% [markdown]
# ### 1.1 MLE vs. Unbiased Estimator for Variance

# %%
# MLE variance estimator (biased): divides by n
var_mle = np.var(diff, ddof=0)
# Unbiased variance estimator: divides by (n-1)
var_unbiased = np.var(diff, ddof=1)

print(f"MLE Variance: {var_mle:.2f}")
print(f"Unbiased Variance: {var_unbiased:.2f}")

# %% [markdown]
# ## 2. MLE vs. Posterior Mode

# %% [markdown]
# Normal model: estimate the mean via MLE and via a Bayesian posterior mode.

# %%
mean_mle = np.mean(diff)
std_mle = np.std(diff, ddof=0)

print(f"MLE Mean: {mean_mle:.2f}")

# Bayesian mode using PyMC
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=diff)
    trace = pm.sample(1000, tune=1000, chains=2, random_seed=123, return_inferencedata=True)

mode_mu = trace.posterior["mu"].mean().values
print(f"Posterior Mean Estimate: {mode_mu:.2f}")

# %% [markdown]
# ## 3. Sampling Distribution Simulation

# %% [markdown]
# Simulate repeated samples to show sampling distributions of mean and variance.

# %%
sample_size = len(diff)
num_samples = 500

means = []
variances = []

for _ in range(num_samples):
    sample = np.random.choice(diff, size=sample_size, replace=True)
    means.append(np.mean(sample))
    variances.append(np.var(sample, ddof=1))

# %% [markdown]
# ## 4. Prior Predictive Sampling with Sharp Prior

# %%
with pm.Model() as sharp_prior_model:
    mu = pm.Normal("mu", mu=mean_mle, sigma=0.1)
    sigma = pm.HalfNormal("sigma", sigma=10)
    y = pm.Normal("y", mu=mu, sigma=sigma)
    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=123)

# %% [markdown]
# ## 5. CI vs. HDI for the Mean

# %%
# Frequentist CI
mean_ci = stats.t.interval(
    0.95,
    df=sample_size-1,
    loc=mean_mle,
    scale=stats.sem(diff)
)
print(f"95% CI (Frequentist): {mean_ci}")

# Bayesian HDI
bayes_hdi = az.hdi(trace.posterior["mu"])
print(f"95% HDI (Bayesian): {bayes_hdi}")

# %% [markdown]
# ## 6. Coverage vs. Probability (Simulated CIs and Posterior HDIs)

# %% [markdown]
# Repeatedly simulate frequentist CIs and overlay Bayesian HDI from the data.

# %%
true_mean = mean_mle
cis = []
for _ in range(100):
    sample = np.random.choice(diff, size=sample_size, replace=True)
    ci = stats.t.interval(0.95, df=sample_size-1, loc=np.mean(sample), scale=stats.sem(sample))
    cis.append(ci)

# Plot CIs and Bayesian HDI
fig, ax = plt.subplots(figsize=(8, 6))
for i, (low, high) in enumerate(cis):
    color = 'green' if (low <= true_mean <= high) else 'red'
    ax.plot([low, high], [i, i], color=color)

ax.axvline(true_mean, color='blue', linestyle='--', label="True Mean")
ax.axvline(bayes_hdi["mu"][0], color='purple', linestyle=':', label="Bayesian HDI")
ax.axvline(bayes_hdi["mu"][1], color='purple', linestyle=':')
ax.legend()
plt.xlabel("Mean")
plt.ylabel("Simulation")
plt.title("Coverage vs. Probability")
plt.show()

# %% [markdown]
# ## 7. Frequentist Linear Regression

# %%
# Baby growth data
baby_url = "https://raw.githubusercontent.com/aloctavodia/BAP3/refs/heads/main/code/data/babies.csv"
babies = pd.read_csv(baby_url)

model = sm.OLS(babies.length, sm.add_constant(babies.month)).fit()
print(model.summary())

# %% [markdown]
# ## 8. Bayesian Linear Regression (Comparison)

# %%
model_bambi = bmb.Model("length ~ month", data=babies)
idata_bambi = model_bambi.fit()
print(idata_bambi)

# %% [markdown]
# ## 9. Summary
# - Frequentist and Bayesian estimators
# - CI vs. HDI
# - Sampling distributions
# - Linear regression comparison
# 
# This draft covers all the key figures and concepts outlined. Further adjustments can be made as needed!
