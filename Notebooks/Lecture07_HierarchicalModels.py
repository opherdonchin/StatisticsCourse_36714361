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
import xarray as xr

# %%
az.style.use("arviz-whitegrid")
plt.rc('figure', dpi=450)

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
pooled_coords = {"data": np.arange(len(tip))}

σ_μ = 5
σ_σ = 1

with pm.Model(coords=pooled_coords) as pooled_model:
    μ = pm.HalfNormal("μ", sigma=σ_μ)
    σ = pm.HalfNormal("σ", sigma=σ_σ)

    y = pm.Gamma("y", mu=μ, sigma=σ, observed=tip, dims="data")


unpooled_coords = {"days": categories, "days_flat":categories[idx]}

with pm.Model(coords=unpooled_coords) as unpooled_model:
    μ = pm.HalfNormal("μ", sigma=σ_μ, dims="days")
    σ = pm.HalfNormal("σ", sigma=σ_σ, dims="days")

    y = pm.Gamma("y", mu=μ[idx], sigma=σ[idx], observed=tip, dims="days_flat")

models = {"pooled": pooled_model, "unpooled": unpooled_model}

# %%
idatas = {}
for model_name, model in models.items():
    with model:
        idata = pm.sample(2000, tune=1000)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        idatas[model_name] = idata



# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for i, (model_name, idata) in enumerate(idatas.items()):
    if model_name == "pooled":
        pm.plot_posterior(idata, var_names=["μ", "σ"], ax=ax[i], hdi_prob='hide', point_estimate=None)
    else:
        for j,day in enumerate(categories):
            pm.plot_posterior(idata, var_names=["μ", "σ"], hdi_prob='hide', point_estimate=None,
                              color=f"C{j}", coords={"days": day}, label=f'{day}', 
                              ax=ax[i])
        ax[i].legend()
    ax[i].set_title(model_name)
plt.show()


# %%
mu = idatas["unpooled"].posterior["μ"]
az.plot_posterior(mu.std(dim="days"))

# %%
gma = pz.Gamma()
pz.maxent(gma, lower=1, upper=4, mass=0.95)
print(f"mu_mu: mean: {gma.mean()}, std: {gma.std()}")

# %%
gma = pz.Gamma()
pz.maxent(gma, lower=0.01, upper=0.5, mass=0.95)
print(f"mu_mu: mean: {gma.mean()}, std: {gma.std()}")

# %%
partial_pooling_coords = {"days": categories, "days_flat":categories[idx]}

μ_μμ = 2.5
σ_μμ = 0.8
μ_σμ = 0.22
σ_σμ = 0.14

with pm.Model(coords=partial_pooling_coords) as partial_pooling_model:
    μ_μ = pm.Gamma("μ_μ", mu=μ_μμ, sigma=σ_μμ)
    σ_μ = pm.Gamma("σ_μ", mu=μ_σμ, sigma=σ_σμ)
    
    μ = pm.Gamma("μ", mu=μ_μ, sigma=σ_μ, dims="days")
    σ = pm.HalfNormal("σ", sigma=1, dims="days")

    y = pm.Gamma("y", mu=μ[idx], sigma=σ[idx], observed=tip, dims="days_flat")

models = {"pooled": pooled_model, "unpooled": unpooled_model, "partial pooling": partial_pooling_model}

# %%
with partial_pooling_model:
    idata = pm.sample(2000, tune=2000, nuts={'target_accept': 0.99})
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
idatas["partial pooling"] = idata

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
for i, (model_name, idata) in enumerate(idatas.items()):
    if model_name == "pooled":
        pm.plot_posterior(idata, var_names=["μ", "σ"], ax=ax[i], hdi_prob='hide', point_estimate=None)
    else:
        for j,day in enumerate(categories):
            pm.plot_posterior(idata, var_names=["μ", "σ"], hdi_prob='hide', point_estimate=None,
                              color=f"C{j}", coords={"days": day}, label=f'{day}', 
                              ax=ax[i])
    ax[i].set_title(model_name)
ax[1].get_legend().remove()
ax[2].legend()
plt.show()

# %%
az.plot_forest([idatas["unpooled"], idatas["partial pooling"]], 
               var_names="μ", model_names=["Unpooled", "Partial pooling"], 
               kind="ridgeplot",
               combined=True, figsize=(6, 4))

# %%
cs_data = pd.read_csv('https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/chemical_shifts_theo_exp.csv')
diff = cs_data.theo - cs_data.exp
cat_encode = pd.Categorical(cs_data['aa'])
idx = cat_encode.codes
coords = {"aa": cat_encode.categories}

# %%
cs_data

# %%
μ_μ = 0
σ_μ = 10
σ_σ = 10

with pm.Model(coords=coords) as cs_p:         
    μ = pm.Normal('μ', mu=μ_μ, sigma=σ_μ) 
    σ = pm.HalfNormal('σ', sigma=σ_σ) 
 
    y = pm.Normal('y', mu=μ, sigma=σ, observed=diff) 
     
    

# %%
pm.model_to_graphviz(cs_p)

# %%
idata_cs_p = pm.sample(random_seed=4591, model=cs_p)

# %%
μ_μ = 0
σ_μ = 10
σ_σ = 10

with pm.Model(coords=coords) as cs_nh:         
    μ = pm.Normal('μ', mu=μ_μ, sigma=σ_μ, dims="aa") 
    σ = pm.HalfNormal('σ', sigma=σ_σ, dims="aa") 
 
    y = pm.Normal('y', mu=μ[idx], sigma=σ[idx], observed=diff) 
     
    

# %%
pm.model_to_graphviz(cs_nh)

# %%
idata_cs_nh = pm.sample(random_seed=4591, model=cs_nh)

# %%

μ_μμ = 0
σ_μμ = 10
σ_σμ = 10

with pm.Model(coords=coords) as cs_h:
    # hyper_priors
    μ_μ = pm.Normal('μ_μ', mu=μ_μμ, sigma=σ_μμ)
    σ_μ = pm.HalfNormal('σ_μ', σ_σμ)

    # priors
    μ = pm.Normal('μ', mu=μ_μ, sigma=σ_μ, dims="aa") 
    σ = pm.HalfNormal('σ', sigma=10, dims="aa") 

    y = pm.Normal('y', mu=μ[idx], sigma=σ[idx], observed=diff) 

    

# %%
pm.model_to_graphviz(cs_h)

# %%
idata_cs_h = pm.sample(random_seed=4591, model=cs_h)

# %%
idata_cs_h

# %%
idata_cs_h.posterior["aa"]

# %%
ax = az.plot_pair(idata_cs_h, var_names=["σ_μ", "μ"], coords={"aa": ["TYR", "PRO"]}, 
             kind="scatter", marginals=False, figsize=(8, 4), textsize=8)


# %%
idata_cs_h.posterior["μ_bar"] = idata_cs_h.posterior["μ"].mean(dim="aa")
idata_cs_h.posterior["s_μ"] = idata_cs_h.posterior["μ"].std(dim="aa")

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
az.plot_pair(idata_cs_h, var_names=["μ_μ", "μ_bar"], kind="scatter", 
             marginals=False, figsize=(8, 4), textsize=8, ax=ax[0])
ax[0].axis('equal')
ax[0].set_title(r"$μ_μ$ and $\frac{1}{N_{aa}}\sum{μ_{aa}}$")

az.plot_pair(idata_cs_h, var_names=["σ_μ", "s_μ"], kind="scatter", 
             marginals=False, figsize=(8, 4), textsize=8, ax=ax[1])
ax[1].axis('equal')
ax[1].set_title(r"$σ_μ$ and $\text{std}({μ_{aa}})$")

# %%
print(f"Pooled model σ_μ: 0")
print(f"Hierarchical model σ_μ: {idata_cs_h.posterior['σ_μ'].mean().values:.2f} ± {idata_cs_h.posterior['σ_μ'].std().values:.2f}")
print(f"Unpooled model σ_μ: 10")
print("")
print(f"Pooled model std(μ): 0")
print(f"Hierarchical model std(μ): {idata_cs_h.posterior['μ'].std(dim='aa').mean().values:.2f} ± {idata_cs_h.posterior['μ'].std(dim='aa').std().values:.2f}")
print(f"Unpooled model std(μ): {idata_cs_nh.posterior['μ'].std(dim='aa').mean().values:.2f} ± {idata_cs_nh.posterior['μ'].std(dim='aa').std().values:.2f}")

# %%
axes = az.plot_forest([idata_cs_nh, idata_cs_h], model_names=['non_hierarchical', 'hierarchical'],
                      var_names='μ', combined=True, r_hat=False, ess=False, figsize=(10, 7),
                      colors='cycle')
y_lims = axes[0].get_ylim()
axes[0].vlines(idata_cs_h.posterior['μ_μ'].mean(), *y_lims, color="k", ls=":");

# %%
football = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/football_players.csv", dtype={'position':'category'})
football

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True, sharey=True)
alphas = [.5, 5., 2.]
betas = [.5, 5., 5.]
for alpha, beta in zip(alphas, betas):
    pz.Beta(alpha, beta).plot_pdf(ax=ax)
ax.set_ylim(0, 5);

fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True, sharey=True)
mus = [0.5, 0.5, 0.286]
sigmas = [0.3536, 0.1507, 0.1598]
for mu, sigma in zip(mus, sigmas):
    pz.Beta(mu=mu, sigma=sigma).plot_pdf(ax = ax)
ax.set_ylim(0, 5);

fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True, sharey=True)
nus = [1.0, 10.0, 7.0]
for mu, nu in zip(mus, nus):
    pz.Beta(mu=mu, nu=nu).plot_pdf(ax=ax)
ax.set_ylim(0, 5);


# %%
## Elciitation for grand mean of success rate
# Define the constraints
lower_bound = 0.0
upper_bound = 0.5
desired_mass = 0.95

dist = pz.Beta()
pz.maxent(dist, lower=lower_bound, upper=upper_bound, mass=desired_mass)
print(f"mu = {dist.mu:.2f}, nu = {dist.nu:.2f}")

# %%

# %%
lower_bound = 10
upper_bound = 300
desired_mass = 0.90

dist = pz.Gamma()
pz.maxent(dist, lower=lower_bound, upper=upper_bound, mass=desired_mass)
print(f"mu = {dist.mu:.2f}, sigma = {dist.sigma:.2f}")

# %%
s1 = pz.Beta(mu=0.25, nu=20).sigma
s2 = pz.Beta(mu=0.25, nu=300).sigma
print(f'Sigma: {s2:.2f} to {s1:.2f}')

# %%
pos_idx = football.position.cat.codes.values
pos_codes = football.position.cat.categories
n_pos = pos_codes.size
n_players = football.index.size

# %%

μ_μμ = 0.22
ν_μμ = 7.4
μ_νμ = 155
σ_νμ = 104

μ_ν = 155
σ_ν = 104

coords = {"pos": pos_codes, "players": np.arange(n_players)}
with pm.Model(coords=coords) as model_football:
    # Hyper parameters
    μ_μ = pm.Beta('μ_μ', mu=μ_μμ, nu=ν_μμ) 
    ν_μ = pm.Gamma('ν_μ', mu=μ_νμ, sigma=σ_νμ)

    
    # Parameters for positions
    μ_p = pm.Beta('μ_p',
                       mu=μ_μ,
                       nu=ν_μ,
                       dims = "pos")
    
    ν_p = pm.Gamma('ν_p', mu=μ_ν, sigma=σ_ν, dims="pos")
 
    # Parameter for players
    θ = pm.Beta('θ', 
                    mu=μ_p[pos_idx],
                    nu=ν_p[pos_idx])
    
    _ = pm.Binomial('gs', n=football.shots.values, p=θ, observed=football.goals.values, dims="players")

# %%
idata_football = pm.sample(draws=2000, model=model_football, target_accept=0.95, random_seed=4591)

# %%
az.summary(idata_football, coords={'θ_dim_0': [0,1,2]})

# %%
ax = [None] * 3

_, ax[0] = plt.subplots(1, 1, figsize=(10, 2), sharex=True)
_, ax[1] = plt.subplots(1, 1, figsize=(10, 2), sharex=True)
_, ax[2] = plt.subplots(1, 1, figsize=(10, 2), sharex=True)

az.plot_posterior(idata_football, var_names='μ_μ', ax=ax[0])
ax[0].set_title(r"Global mean")
ax[0].set_xlim([0.02, 0.23])
az.plot_posterior(idata_football.posterior.sel(pos="FW"), var_names='μ_p', ax=ax[1])
ax[1].set_title(r"Forward position mean")
ax[1].set_xlim([0.02, 0.23])
az.plot_posterior(idata_football.posterior.sel(θ_dim_0=1457), var_names='θ', ax=ax[2])
ax[2].set_title(r"Messi mean")
ax[2].set_xlim([0.02, 0.23])


# %%
N = 20
groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
M = len(groups)
idx = np.repeat(range(M - 1), N)
idx = np.append(idx, 7)
np.random.seed(314)
alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(6, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))
y_m = np.zeros(len(idx))
x_m = np.random.normal(0, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real
_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
ax = np.ravel(ax)
j, k = 0, N
for i, g in enumerate(groups):
    ax[i].scatter(x_m[j:k], y_m[j:k], marker=".")
    ax[i].set_title(f"group {g}")

    j += N
    k += N

# %%
coords = {"group": groups, "data": np.arange(len(y_m))}

with pm.Model(coords=coords) as unpooled_model:
    α = pm.Normal("α", mu=0, sigma=10, dims="group")
    β = pm.Normal("β", mu=0, sigma=10, dims="group")
    σ = pm.HalfNormal("σ", 5)
    _ = pm.Normal("y_pred", mu=α[idx] + β[idx] * x_m, sigma=σ, observed=y_m, dims="data")

# %%
idata_up = pm.sample(random_seed=123, model=unpooled_model)

# %%

alpha_samples = idata_up.posterior["α"]  # shape (sample, group)
beta_samples  = idata_up.posterior["β"]

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.ravel()

for i, group_name in enumerate(groups):
    mask = (idx == i)
    x_i = x_m[mask]
    y_i = y_m[mask]
    
    ax = axes[i]
    ax.scatter(x_i, y_i, alpha=0.6)
    ax.set_title(f"Group {group_name}")

    x_grid = xr.DataArray(np.linspace(x_i.min() - 0.5, x_i.max() + 0.5, 100), dims=["plot_data"], name="x_grid")
    alpha_i = alpha_samples.isel(group=i)  # shape (sample,)
    beta_i  = beta_samples.isel(group=i)
    y_hat = alpha_i + beta_i * x_grid
    y_mean = y_hat.mean(dim=["chain", "draw"])  # shape (len(x_grid),)

    hdi = az.hdi(y_hat, hdi_prob=0.94)  # shape (len(x_grid), 2)

    ax.plot(x_grid, y_mean, color="C1", lw=2, label="Mean")
    ax.fill_between(
        x_grid,
        hdi['x'].sel(hdi='lower'),
        hdi['x'].sel(hdi='higher'),
        color="C1",
        alpha=0.3,
        label="HDI"
    )

    if i==0:
        ax.legend()
    
    ax.set_ylim([-1.6, 6.6])


plt.tight_layout()
plt.show()


# %%
μ_αμ = y_m.mean()
σ_αμ = 1
σ_ασ = 5
β_αμ = 1
β_ασ = 5
σ_σ = 5

with pm.Model(coords=coords) as hierarchical_centered:
    # hyper-priors
    α_μ = pm.Normal("α_μ", mu=μ_αμ, sigma=σ_αμ)
    α_σ = pm.HalfNormal("α_σ", σ_ασ)
    β_μ = pm.Normal("β_μ", mu=0, sigma=β_αμ)
    β_σ = pm.HalfNormal("β_σ", sigma=β_ασ)

    # priors
    α = pm.Normal("α", mu=α_μ, sigma=α_σ, dims="group")
    β = pm.Normal("β", mu=β_μ, sigma=β_σ, dims="group")
    σ = pm.HalfNormal("σ", σ_σ)
    _ = pm.Normal("y_pred", mu=α[idx] + β[idx] * x_m, sigma=σ, observed=y_m, dims="data")


# %%
idata_cen = pm.sample(random_seed=123, model=hierarchical_centered)

# %%
az.summary(idata_cen, var_names=["α_μ", "α_σ", "β_μ", "β_σ", "σ"], hdi_prob=0.94)

# %%
az.summary(idata_cen, var_names=["β", "β_μ"])

# %%
idata_cen.posterior["log(β_σ)"] = np.log(idata_cen.posterior["β_σ"])

plt.figure()
az.plot_pair(idata_cen, var_names=["log(β_σ)", "β"], coords={'group': ['H']}, divergences=True, kind="scatter")

plt.figure()
az.plot_pair(idata_cen, var_names=["log(β_σ)", "β_μ", "β"], coords={'group': ['G']}, divergences=True, kind="scatter")


# %%
with pm.Model(coords=coords) as hierarchical_non_centered:
    # hyper-priors
    α_μ = pm.Normal("α_μ", mu=μ_αμ, sigma=σ_αμ)
    α_σ = pm.HalfNormal("α_σ", σ_ασ)
    β_μ = pm.Normal("β_μ", mu=0, sigma=β_αμ)
    β_σ = pm.HalfNormal("β_σ", sigma=β_ασ)

    # priors
    α = pm.Normal("α", mu=α_μ, sigma=α_σ, dims="group")

    β_offset = pm.Normal("β_offset", mu=0, sigma=1, dims="group")
    β = pm.Deterministic("β", β_μ + β_offset * β_σ, dims="group")

    σ = pm.HalfNormal("σ", σ_σ)
    _ = pm.Normal("y_pred", mu=α[idx] + β[idx] * x_m, sigma=σ, observed=y_m, dims="data")

# %%
idata_ncen = pm.sample(random_seed=123, target_accept=0.85, model=hierarchical_non_centered)

# %%
az.summary(idata_ncen, var_names=["α_μ", "α_σ", "β_μ", "β_σ", "σ"], hdi_prob=0.94)

# %%
idata_ncen.posterior["log(β_σ)"] = np.log(idata_ncen.posterior["β_σ"])

plt.figure()
az.plot_pair(idata_ncen, var_names=["log(β_σ)", "β"], coords={'group': ['H']}, divergences=True, kind="scatter")

plt.figure()
az.plot_pair(idata_ncen, var_names=["log(β_σ)", "β_offset", "β"], coords={'group': ['H']}, divergences=True, kind="scatter")

# %%

alpha_samples = idata_ncen.posterior["α"]  # shape (sample, group)
beta_samples  = idata_ncen.posterior["β"]

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.ravel()

for i, group_name in enumerate(groups):
    mask = (idx == i)
    x_i = x_m[mask]
    y_i = y_m[mask]
    
    ax = axes[i]
    ax.scatter(x_i, y_i, alpha=0.6)
    ax.set_title(f"Group {group_name}")

    x_grid = xr.DataArray(np.linspace(x_i.min() - 0.5, x_i.max() + 0.5, 100), dims=["plot_data"], name="x_grid")
    alpha_i = alpha_samples.isel(group=i)  # shape (sample,)
    beta_i  = beta_samples.isel(group=i)
    y_hat = alpha_i + beta_i * x_grid
    y_mean = y_hat.mean(dim=["chain", "draw"])  # shape (len(x_grid),)

    hdi = az.hdi(y_hat, hdi_prob=0.94)  # shape (len(x_grid), 2)

    ax.plot(x_grid, y_mean, color="C1", lw=2, label="Mean")
    ax.fill_between(
        x_grid,
        hdi['x'].sel(hdi='lower'),
        hdi['x'].sel(hdi='higher'),
        color="C1",
        alpha=0.3,
        label="HDI"
    )
    ax.set_ylim([-1.6, 6.6])

    if i==0:
        ax.legend()

plt.tight_layout()
print(ax.get_ylim())

plt.show()

