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

# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz


# %%
az.style.use("arviz-whitegrid")
plt.rc('figure', dpi=450)

# %%
data = np.loadtxt("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/chemical_shifts.csv")

# %%
_, ax = plt.subplots(figsize=(12, 3))
box = ax.boxplot(data, patch_artist=True, vert=False)

# Extracting the graphic objects for the legend
median = box['medians'][0]
whiskers = box['whiskers'][0]
fliers = box['fliers'][0]
boxes = box['boxes'][0]

# Creating the legend
ax.legend([boxes, whiskers, fliers], ['IQR', 'Whiskers', 'Outliers'], loc='upper right')

plt.show()

# %%

# Removing outliers from the boxplot
_, ax = plt.subplots(figsize=(12, 3))
box = ax.boxplot(data, patch_artist=True, vert=False, showfliers=False)

# Adding the data points with jitter on the y-axis
y = np.random.normal(1, 0.02, size=len(data))
data_points = ax.plot(data, y, 'ro', alpha=0.5, label="Data")

# Extracting the graphic objects for the legend
median = box['medians'][0]
whiskers = box['whiskers'][0]
boxes = box['boxes'][0]

# Creating the legend
ax.legend([data_points[0], boxes, whiskers], ['Data', 'IQR', 'Whiskers'], loc='upper right')



plt.show()


# %%
mu = [0, 0, 2]
sigma = [1, 0.5, 1]
_, ax = plt.subplots(figsize=(6, 3))
for i in range(3):
    normal_distribution = pz.Normal(mu=mu[i], sigma=sigma[i])
    normal_distribution.plot_pdf(ax=ax)

plt.show()

# %%
l = [0.0, 49.0, 55.0]
h = [100.0, 70.0, 65.0]
_, ax = plt.subplots(figsize=(6, 3))
for i in range(3):
    uniform_dist = pz.Uniform(lower=l[i], upper=h[i])
    uniform_dist.plot_pdf(ax=ax)
    
plt.show()

# %%
sigma = [2.0, 5.0, 10.0]

_, ax = plt.subplots(figsize=(6, 3))
for i in range(3):
    halfnormal_dist = pz.HalfNormal(sigma=sigma[i])
    halfnormal_dist.plot_pdf(ax=ax)
    
plt.show()

# %%
l_mu = 40
h_mu = 70
σ_σ = 5


# %%
with pm.Model() as model_g:
    μ = pm.Uniform('μ', lower=l_mu, upper=h_mu)
    σ = pm.HalfNormal('σ', sigma=σ_σ)
    y = pm.Normal('y', mu=μ, sigma=σ, observed=data)


# %%
# Sample from the prior distribution
with model_g:
    idata_g = pm.sample_prior_predictive(samples=1000, random_seed=42)
idata_g

# %%
plt.figure(figsize=(12, 6))
ax = az.plot_ppc(idata_g, group='prior', num_pp_samples=1, mean=False, observed=True, random_seed=14)
ax.get_lines()[2].set_alpha(1.0)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Prior Predictive from model_g')

plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 6))
ax = az.plot_ppc(idata_g, group='prior', num_pp_samples=4, mean=False, observed=True, random_seed=14)

for l in ax.get_lines()[2:]:
    l.set_alpha(0.9)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Prior Predictive from model_g')

plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 6))
ax = az.plot_ppc(idata_g, group='prior', num_pp_samples=20, mean=False, observed=True, random_seed=14)

for l in ax.get_lines()[2:]:
    l.set_alpha(0.9)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Prior Predictive from model_g')

plt.legend()
plt.show()

# %%
with model_g:
    idata_g.extend(pm.sample(random_seed=4591), join='right')

# %%
idata_g

# %%
az.plot_trace(idata_g, compact=False);

# %%
az.plot_pair(idata_g, kind='scatter', marginals=True);

# %%
az.plot_pair(idata_g, kind='kde', marginals=True);

# %%
az.summary(idata_g, kind='stats', round_to=2)

# %%
pm.sample_posterior_predictive(idata_g, model=model_g, extend_inferencedata=True, random_seed=4591)

# %%
az.plot_ppc(idata_g, num_pp_samples=100, figsize=(10, 4), colors=["C1", "C0", "C1"])

# %%
for nu in [1, 2, 10]:
    pz.StudentT(nu, 0, 1).plot_pdf(support=(-5, 5), figsize=(12, 4))

ax = pz.StudentT(np.inf, 0, 1).plot_pdf(support=(-5, 5), figsize=(12, 4), color="k")
ax.get_lines()[-1].set_linestyle("--")
pz.internal.plot_helper.side_legend(ax)

# %%
for nu in [1, 2, 10]:
    pz.StudentT(nu, 0, 1).plot_pdf(support=(-7, 7), figsize=(12, 4))

ax = pz.StudentT(np.inf, 0, 1).plot_pdf(support=(-7, 7), figsize=(12, 4), color="k")
ax.get_lines()[-1].set_linestyle("--")
pz.internal.plot_helper.side_legend(ax)
ax.set_ylim(0, 0.07)
plt.show()

# %%
l_mu = 40
h_mu = 70
σ_σ = 10
λ_ν = 1/30

# %%
with pm.Model() as model_t:
    μ = pm.Uniform('μ', l_mu, h_mu)
    σ = pm.HalfNormal('σ', sigma=σ_σ)
    ν = pm.Exponential('ν', λ_ν)
    y = pm.StudentT('y', nu=ν, mu=μ, sigma=σ, observed=data)

# %%
idata_t = pm.sample(random_seed=4591, model=model_t)

# %%
az.plot_trace(idata_t, compact=False);

# %%
az.summary(idata_t, kind="stats", round_to=2)

# %%
ax

# %%
ax = az.plot_pair(idata_t, kind='kde', marginals=True)
for row in ax:
	for subplot in row:
		if subplot is not None:
			subplot.set_xlabel(subplot.get_xlabel(), fontsize=30)
			subplot.set_ylabel(subplot.get_ylabel(), fontsize=30)
			subplot.tick_params(axis='both', which='major', labelsize=26)

# %%
ax = az.plot_pair(idata_t, var_names=['σ', 'ν'], kind='kde', marginals=False)

ax.set_xlabel(ax.get_xlabel(), fontsize=18)
ax.set_ylabel(ax.get_ylabel(), fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlim(1.2, 3.0)
ax.set_ylim(0, 15)

# %%
pm.sample_posterior_predictive(idata_t, model=model_t, extend_inferencedata=True, random_seed=4591)

# %%
ax = az.plot_ppc(idata_t, num_pp_samples=100, figsize=(10, 4), colors=["C1", "C0", "C1"])
ax.set_xlim(39, 70)

# %%
tips = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data//tips.csv")
tips.tail()

# %%
_, ax = plt.subplots(figsize=(10, 4))
az.plot_forest(tips.pivot(columns="day", values="tip").to_dict("list"),
               kind="ridgeplot",
               hdi_prob=1,
               ax=ax)
ax.set_xlabel("Tip amount")
ax.set_ylabel("Day")

# %%
categories = np.array(["Thur", "Fri", "Sat", "Sun"])

tip = tips["tip"].values
idx = pd.Categorical(tips["day"], categories=categories).codes

# %%
coords = {"days": categories, "days_flat":categories[idx]}

with pm.Model(coords=coords) as comparing_groups:
    μ = pm.HalfNormal("μ", sigma=5, dims="days")
    σ = pm.HalfNormal("σ", sigma=1, dims="days")

    y = pm.Gamma("y", mu=μ[idx], sigma=σ[idx], observed=tip, dims="days_flat")

# %%
idata_cg = pm.sample(random_seed=4591, model=comparing_groups)
idata_cg.extend(pm.sample_posterior_predictive(idata_cg, random_seed=4591, model=comparing_groups))

# %%
_, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
az.plot_ppc(idata_cg, num_pp_samples=100,
            colors=["C0", "C4", "C1"],
            coords={"days_flat":[categories]}, flatten=[], ax=axes)

# %%
cg_posterior = az.extract(idata_cg)

dist = pz.Normal(0, 1)

comparisons = [("Fri", "Sun"), ("Sat", "Sun")]

_, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

for (i, j), ax in zip(comparisons, axes.ravel()):
    means_diff = cg_posterior["μ"].sel(days=i) - cg_posterior['μ'].sel(days=j)
    
    az.plot_posterior(means_diff.values, hdi_prob='hide', ref_val=None, ax=ax, label="Posterior of difference")

    ax.set_xlabel("Difference of means")
    ax.set_title(f"{i} - {j}")

# %%
cg_posterior = az.extract(idata_cg)

dist = pz.Normal(0, 1)

comparisons = [("Fri", "Sun"), ("Sat", "Sun")]

_, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

for (i, j), ax in zip(comparisons, axes.ravel()):
    means_diff = cg_posterior["μ"].sel(days=i) - cg_posterior['μ'].sel(days=j)
    
    az.plot_posterior(means_diff.values, hdi_prob='hide', ref_val=None, ax=ax, label="Posterior of difference")
    ax.axvline(0, color="C1", linestyle="--", lw=1)
    ax.set_xlabel("Difference of means")
    ax.set_title(f"{i} - {j}")

# %%
cg_posterior = az.extract(idata_cg)

dist = pz.Normal(0, 1)

comparisons = [("Fri", "Sun"), ("Sat", "Sun")]

_, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

for (i, j), ax in zip(comparisons, axes.ravel()):
    means_diff = cg_posterior["μ"].sel(days=i) - cg_posterior['μ'].sel(days=j)
    
    az.plot_posterior(means_diff.values, hdi_prob='hide', ref_val=0.0, ax=ax, label="Posterior of difference")
    ax.set_xlabel("Difference of means")
    ax.set_title(f"{i} - {j}")

# %%
cg_posterior = az.extract(idata_cg)

dist = pz.Normal(0, 1)

comparisons = [("Fri", "Sun"), ("Sat", "Sun")]

_, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

for (i, j), ax in zip(comparisons, axes.ravel()):
    means_diff = cg_posterior["μ"].sel(days=i) - cg_posterior['μ'].sel(days=j)
    
    az.plot_posterior(means_diff.values, hdi_prob=0.94, ref_val=0.0, ax=ax,)
    ax.set_xlabel("Difference of means")
    ax.set_title(f"{i} - {j}")

# %%
cg_posterior = az.extract(idata_cg)

dist = pz.Normal(0, 1)

comparisons = [("Fri", "Sun"), ("Sat", "Sun")]

_, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

for (i, j), ax in zip(comparisons, axes.ravel()):
    means_diff = cg_posterior["μ"].sel(days=i) - cg_posterior['μ'].sel(days=j)
    
    az.plot_posterior(means_diff.values, hdi_prob=0.94, ref_val=0.0, rope=[-0.1, 0.1], ax=ax,)
    ax.set_xlabel("Difference of means")
    ax.set_title(f"{i} - {j}")

# %%
cg_posterior = az.extract(idata_cg)

dist = pz.Normal(0, 1)

comparisons = [(categories[i], categories[j]) for i in range(4) for j in range(i+1, 4)]

_, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)

for (i, j), ax in zip(comparisons, axes.ravel()):
    means_diff = cg_posterior["μ"].sel(days=i) - cg_posterior['μ'].sel(days=j)
    
    d_cohen = (means_diff /
               np.sqrt((cg_posterior["σ"].sel(days=i)**2 + 
                        cg_posterior["σ"].sel(days=j)**2) / 2)
              ).mean().item()
    
    ps = dist.cdf(d_cohen/(2**0.5))
    az.plot_posterior(means_diff.values, ref_val=0, ax=ax)
    ax.set_title(f"{i} - {j}")
    ax.plot(0, label=f"Cohen's d = {d_cohen:.2f}\nProb sup = {ps:.2f}", alpha=0)
    ax.legend(loc=1)
