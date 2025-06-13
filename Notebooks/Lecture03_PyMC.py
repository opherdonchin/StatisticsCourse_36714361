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

import pytensor.printing
from pytensor.graph.basic import ancestors
from pytensor.d3viz import d3viz
import matplotlib.pyplot as plt
from pytensor.printing import pydotprint

# %%
az.style.use("arviz-whitegrid")
plt.rc('figure', dpi=450)

# %%
# Find Beta distribution parameters for mode = 0.7 and 95% HDI from 0.49 to 0.86
target_mode = 0.7
target_lower_bound = 0.49
target_upper_bound = 0.86

# Use Preliz to find the parameters
beta_dist = pz.Beta()
ax, beta_params = pz.maxent(beta_dist, mass=0.95, 
                        lower=target_lower_bound, upper=target_upper_bound, plot=True)

# Display the found parameters
print(f"Beta distribution parameters: alpha = {beta_params['x'][0]:.4f}, beta = {beta_params['x'][1]:.4f}")


# %%
# Extract the parameters from beta_params
alpha = 15  # Approximately 15.31
beta = 7   # Approximately 7.22

# Create a figure and axis
fig, ax = plt.subplots(figsize=(9/2.54, 7/2.54))

# Generate the Beta distribution plot with arviz
x = np.linspace(0, 1, 1000)
dist = pz.Beta(alpha=alpha, beta=beta)

dist.plot_pdf(ax=ax, legend=None)
# Calculate the actual mode of the Beta distribution
actual_mode = (alpha - 1) / (alpha + beta - 2)

# Add the mode to the plot
ax.axvline(actual_mode, linestyle='--', color='red', alpha=0.7)
ax.text(actual_mode + 0.03, 2.5, f"mode = {actual_mode:.2f}", 
    color='red', fontsize=12, va='center')

# Add labels and title
ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.set_title(f'Beta({alpha:.0f}, {beta:.0f})')

# Set limits and adjust layout
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

plt.tight_layout()



# %%
sample_sizes = [500, 5000, 50000]

for size in sample_sizes:
    # Sample from the Beta distribution
    samples = dist.rvs(size=size, random_state=123)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(9/2.54, 7/2.54))

    # Plot histogram using ArviZ (with density normalization)
    az.plot_dist(samples, ax=ax, kind='hist', color='blue')

    # Estimate KDE to find the sample mode
    kde_x, kde_y = az.kde(samples)
    sample_mode = kde_x[np.argmax(kde_y)]

    # Add mode line and text
    ax.axvline(sample_mode, linestyle='--', color='red', alpha=0.7)
    ax.text(sample_mode + 0.03, max(kde_y)*0.9, f"mode = {sample_mode:.2f}", 
            color='red', fontsize=12, va='center')

    # Labels and limits
    ax.set_xlabel('θ')
    ax.set_ylabel('Density')
    ax.set_title(f'Number of samples: {size}')
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # Layout
    plt.tight_layout()
    plt.show()



# %% [markdown]
# ```stan
# #include "./stan_example.stan"

# %%
N = 100
y = np.random.normal(0, 1, N)
mu_mu = 0
sd_mu = 1
sd_sd = 1

coords = {"data": np.arange(N)}
with pm.Model(coords=coords) as example_model:
    mu = pm.Normal("mu", mu=mu_mu, sigma=sd_mu)
    sd = pm.Normal("sd", mu=0, sigma=sd_sd)
    y = pm.Normal("y", mu=mu, sigma=sd, observed=y)

# %%
np.random.seed(123)
trials = 4
theta_real = 0.35 # unknown value in a real experiment
data = pz.Binomial(n=1, p=theta_real).rvs(trials)
data

# %%
with pm.Model() as our_first_model:
    theta = pm.Beta('theta', alpha=1., beta=1.)
    y = pm.Bernoulli('y', p=theta, observed=data)

# %%
idata = pm.sample(1000, model=our_first_model, random_seed=4591)

# %%
idata

# %%
az.plot_trace(idata, chain_prop='color')

# %%
pm.model_to_graphviz(our_first_model)

# %%
our_first_model.basic_RVs

# %%
pytensor.dprint(y)

# %%
# Generate the graph image data without writing to disk
img_data = pydotprint(y, format="png", var_with_name_simple=True, return_image=True)

# Display the image in a Jupyter/Colab notebook
from IPython.display import Image, display
display(Image(img_data))

# %%
for i in range(10):
    print(f"Pytensor sample {i}: {theta.eval()}")

# %%
for i in range(10):
    print(f"Sample {i}: {pm.draw(theta)}")

# %%
for i in range(10):
    random_theta, random_y = pm.draw([theta, y])
    print(f"Sample {i}: theta = {random_theta}, y = {random_y}")


# %%
from scipy import special 

sample_theta  = pm.draw(theta)    

logp_func = our_first_model.compile_logp()
joint_logp = logp_func({"theta_logodds__": special.logit(sample_theta)})

print("Sampled theta:", sample_theta)
print("Sampled y:", data)
print("Joint log probability:", joint_logp)
