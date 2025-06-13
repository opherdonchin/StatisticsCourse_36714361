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

# %% [markdown] id="hYmV2-_S8l-r"
# # **Plot BPV in low data regimes**
#

# %% [markdown] id="H1t-wLOn8okL"
# ### Imports

# %% colab={"base_uri": "https://localhost:8080/"} id="cHoY1VtY8hqi" outputId="f6ea01d6-b36f-43ad-a6e1-52f52d513892"
import numpy as np # arrays, array operations
import scipy.stats as stats # statistics
import matplotlib.pyplot as plt # plot graphs
import pandas as pd #dataframes
import io
import xarray as xr #multidimensional dataframes
import pymc as pm
import arviz as az
import scipy.interpolate as interpolate
import preliz

# %% [markdown] id="ECxi4lYOBGmz"
# ### Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 108} id="RDDb7Wb8BIeL" outputId="ca56dd4d-990e-47cc-a77a-9c22538af073"

data = pd.read_csv('..\data\muscle_force.csv')
for col in data.columns: # see names of the columns in a loop
    print(col)

# %% [markdown] id="Kr1ZGpXGBQ9t"
# Plot

# %% colab={"base_uri": "https://localhost:8080/", "height": 454} id="JHZtM9T5BUSr" outputId="146d6f63-1adf-4c84-aeb8-50b997a666ef"
plt.scatter(data.Length, data.Force)

plt.xlabel('Length [cm]', fontsize = 14)
plt.ylabel('Force [N]', fontsize = 14)


# %% [markdown] id="0muqRaMJBw5J"
# ### Simple Linear Regression

# %% [markdown] id="xq3yWGb0GeyT"
# Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 263, "referenced_widgets": ["dc8acdd670dc40219be1dd9de9d1a93b", "4bc4de62b39a4de0b704ab1efd7866d9"]} id="HSti245xBzn7" outputId="3558ae1b-1e60-4041-93d1-a9e596bdea63"
coords = {"data": np.arange(len(data))}
with pm.Model(coords=coords) as model_slr:
    b0 = pm.Normal("b0", mu=50, sigma=50)
    b1 = pm.Normal("b1", mu=0, sigma=50)
    sig = pm.HalfNormal("sig", 10)
    mu = pm.Deterministic("mu", b0 + b1 * data.Length, dims="data")
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sig, observed=data.Force, dims="data")

    idata_slr = pm.sample(1000, chains = 4)

idata_slr

# %% [markdown] id="XFkFn0gwGf_6"
# Posterior Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 320} id="JONm6pgdGhQx" outputId="c229b986-c062-4d55-d953-9146976f5330"
az.plot_posterior(idata_slr, var_names=["~mu"], figsize=(12, 3))

# %% colab={"base_uri": "https://localhost:8080/", "height": 460} id="dmHcWZH7JjVR" outputId="250cf764-2bf9-40e4-d9cf-6b92ad050f29"
#pair plot
az.plot_pair(idata_slr, var_names=['b0', 'b1'])

# %% [markdown] id="6J0par_yQXe0"
# Posterior Predictive

# %% colab={"base_uri": "https://localhost:8080/", "height": 177, "referenced_widgets": ["2f34c514e9e14cc6be2fecba6651f00d", "f9c1061ee941401eaea853708e0cad3c"]} id="5-V5VScuQY_T" outputId="c73632a0-3cd3-44c9-c3f3-517417c7046b"
pm.sample_posterior_predictive(idata_slr, model=model_slr, extend_inferencedata=True)


# %% [markdown] id="CyIIj4DWg4Tw"
# Bayesian p-value

# %% colab={"base_uri": "https://localhost:8080/", "height": 454} id="fVsicNBJg6cE" outputId="f80b0008-7d61-42bd-a912-e1ed27234a9f"
#distribution
az.plot_bpv(idata_slr, kind="p_value")


# %% [markdown]
# ### Resample the data
#

# %%
data_resampled = data.sample(n=100, replace=True)
data_resampled

# %%
coords = {"data": np.arange(len(data_resampled))}
with pm.Model(coords=coords) as model_resampled:
    b0 = pm.Normal("b0", mu=50, sigma=50)
    b1 = pm.Normal("b1", mu=0, sigma=50)
    sig = pm.HalfNormal("sig", 10)
    mu = pm.Deterministic("mu", b0 + b1 * data_resampled.Length, dims="data")
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sig, observed=data_resampled.Force, dims="data")

    idata_resampled = pm.sample(1000, chains = 4)

idata_resampled

# %%
pm.sample_posterior_predictive(idata_resampled, model=model_resampled, extend_inferencedata=True)

# %%
az.plot_bpv(idata_resampled, kind="p_value")
