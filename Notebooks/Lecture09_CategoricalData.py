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
import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import bambi as bmb
import xarray as xr

from ipywidgets import interact
import ipywidgets as ipyw

# %%
penguins = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/penguins.csv").dropna()
penguins.head()

# %%
# %matplotlib inline
# Get unique categories
unique_categories = np.unique(penguins.species)

# Create color map for categories
category_color_map = {cat: f"C{i}" for i, cat in enumerate(unique_categories)}

# Generate colors for each category
colors = [category_color_map[cat] for cat in penguins.species]

# Create scatter plot for each category
for cat in unique_categories:
    category_data = penguins[penguins.species == cat]
    plt.scatter(category_data.body_mass, category_data.bill_length, c=category_color_map[cat], label=cat)

# Add labels and legend
plt.ylabel("Body mass (g)")
plt.xlabel("Bill length (mm)")
plt.legend(labels=unique_categories, loc="lower right", title="Species")
plt.title("Penguin body  mass as a function of bill length across species")
plt.show()

# %%
coords = {"data": np.arange(len(penguins))}

with pm.Model(coords=coords) as model_pooled_pm:
    β0 = pm.Normal("β0", mu=0, sigma=10)
    β1 = pm.Normal("β1", mu=0, sigma=10)
    σ = pm.HalfNormal("σ", sigma=1)

    μ = pm.Deterministic("μ", β0 + β1 * penguins.bill_length, dims="data")
    y_obs = pm.Normal("y_obs", mu=μ, sigma=σ, observed=penguins.body_mass, dims="data")

# %%
idata_pooled_pm = pm.sample(model=model_pooled_pm, random_seed=123)

# %%
x_grid = xr.DataArray(
    np.linspace(penguins.bill_length.min(), penguins.bill_length.max(), 100),
    dims="plot_x"
)
posterior = idata_pooled_pm.posterior
μ_samples = (
    posterior["β0"] + posterior["β1"] * x_grid
)
fig, ax = plt.subplots(figsize=(10, 6))

for cat in unique_categories:
    category_data = penguins[penguins.species == cat]
    plt.scatter(category_data.body_mass, category_data.bill_length, c=category_color_map[cat], label=cat)

mean_μ = μ_samples.mean(dim=("chain", "draw"))
ax.plot(x_grid, mean_μ, color="C0")

# Plot the HDI band
az.plot_hdi(x_grid, μ_samples, hdi_prob=0.94, color="C0", smooth=False, fill_kwargs={"alpha":0.3}, ax=ax)

ax.set_xlabel("Bill Length")
ax.set_ylabel("Body Mass")
ax.legend()
plt.show()


# %%
model_pooled_bmb = bmb.Model("body_mass ~ bill_length", data=penguins)
idata_pooled_bmb = model_pooled_bmb.fit(random_seed=123)

# %%
idata_pooled_bmb

# %%
x_grid = xr.DataArray(
    np.linspace(penguins.bill_length.min(), penguins.bill_length.max(), 100),
    dims="plot_x"
)
posterior = idata_pooled_bmb.posterior
μ_samples = (
    posterior["Intercept"] + posterior["bill_length"] * x_grid
)
fig, ax = plt.subplots(figsize=(10, 6))

for cat in unique_categories:
    category_data = penguins[penguins.species == cat]
    plt.scatter(category_data.body_mass, category_data.bill_length, c=category_color_map[cat], label=cat)

mean_μ = μ_samples.mean(dim=("chain", "draw"))
ax.plot(x_grid, mean_μ, color="C0")

# Plot the HDI band
az.plot_hdi(x_grid, μ_samples, hdi_prob=0.94, color="C0", smooth=False, fill_kwargs={"alpha":0.3}, ax=ax)

ax.set_xlabel("Bill Length")
ax.set_ylabel("Body Mass")
ax.legend()
plt.show()


# %%
species_codes, species_categories = pd.factorize(penguins['species'])
penguins['species_code'] = species_codes

coords = {"data": np.arange(len(penguins)), "species": species_categories}

with pm.Model(coords=coords) as model_unpooled_pm:
    β0 = pm.Normal("β0", mu=0, sigma=10)
    β1 = pm.Normal("β1", mu=0, sigma=10)
    β_species = pm.Normal("β_species", mu=0, sigma=10, dims="species")
    σ = pm.HalfNormal("σ", sigma=1)

    μ = pm.Deterministic(
        "μ", 
        β0 + β1 * penguins.bill_length.values + β_species[species_codes], 
        dims="data"
    )
    y_obs = pm.Normal("y_obs", mu=μ, sigma=σ, observed=penguins.body_mass, dims="data")

# %%
idata_unpooled_pm = pm.sample(model=model_unpooled_pm, random_seed=123)

# %%
idata_unpooled_pm

# %%
# Define x-grid
x_grid = xr.DataArray(
    np.linspace(penguins.bill_length.min(), penguins.bill_length.max(), 100),
    dims="plot_x"
)

posterior = idata_unpooled_pm.posterior  # Make sure this points to the correct InferenceData object

fig, ax = plt.subplots(figsize=(10, 6))

# Loop over species
for cat in species_categories:
    # Scatter data points for the species
    category_data = penguins[penguins.species == cat]
    ax.scatter(
        category_data.bill_length, 
        category_data.body_mass, 
        color=category_color_map[cat], 
        alpha=0.5, 
        label=cat
    )

    # Get the species index
    cat_index = np.where(species_categories == cat)[0][0]
    
    # Calculate μ_samples per species
    μ_samples = (
        posterior["β0"] +
        posterior["β1"] * x_grid +
        posterior["β_species"][:, :, cat_index]
    )
    
    # Mean line
    mean_μ = μ_samples.mean(dim=("chain", "draw"))
    ax.plot(x_grid, mean_μ, color=category_color_map[cat])
    
    # HDI band
    az.plot_hdi(
        x_grid, μ_samples, hdi_prob=0.94, 
        color=category_color_map[cat], 
        fill_kwargs={"alpha":0.2}, 
        ax=ax
    )

ax.set_xlabel("Bill Length")
ax.set_ylabel("Body Mass")
ax.legend()
plt.show()

# %%
model_unpooled_bmb = bmb.Model("body_mass ~ bill_length + species", data=penguins)
idata_unpooled_bmb = model_unpooled_bmb.fit(random_seed=123)

# %%
idata_unpooled_bmb

# %%
ax = az.plot_forest(idata_p, combined=True, figsize=(6, 3));
mean_chinstrap = idata_p.posterior["species"].sel(species_dim="Chinstrap").mean()
mean_gentoo = idata_p.posterior["species"].sel(species_dim="Gentoo").mean()
ax[0].annotate(f"{mean_chinstrap.item():.2f}", (mean_chinstrap , 1.15), weight='bold', horizontalalignment='center')
ax[0].annotate(f"{mean_gentoo.item():.2f}", (mean_gentoo , 0.4), weight='bold', horizontalalignment='center')

# %%
bmb.interpret.plot_predictions(model_p, idata_p, ["bill_length",  "species"], fig_kwargs={"figsize":(6, 6)})

# %%
model_p.graph()

# %%
model_p_pp = bmb.Model("body_mass ~(bill_length|species)", data=penguins)
model_p_pp.build()
model_p_pp.graph()

# %%
model_p_pps = bmb.Model("body_mass ~ (0 + bill_length|species)", data=penguins)
model_p_pps.build()
model_p_pps.graph()

# %%
# #%%javascript
#IPython.OutputArea.prototype._should_scroll = function(lines) {return false}

# %%
# %matplotlib widget
x = np.linspace(-10, 10)

X0, X1 = np.meshgrid(x, x)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=-45)
#ax.view_init(elev=0, azim=-90)


def update_lm(b0, b1, b2):
    ax.clear()
    Y = X0 * b0 + X1 * b1 + b2 * (X0 * X1) 
    line = ax.plot_wireframe(X0, X1, Y, rstride=5, cstride=5)
    ax.set_zlim(-100, 100)
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.set_zticks([-100, -50, 0, 50, 100])
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('μ')


interact(update_lm,
         b0=ipyw.FloatSlider(min=-5, max=5, step=0.1, value=0),
         b1=ipyw.FloatSlider(min=-5, max=5, step=0.1, value=0),
         b2=ipyw.FloatSlider(min=-1, max=1, step=0.1, value=0));
