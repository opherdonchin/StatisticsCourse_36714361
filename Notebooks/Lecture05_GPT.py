# Lecture 5 Notebook Generator: MCMC

# %% [markdown]
# # Lecture 5 Notebook: MCMC
# This notebook generates all figures used in Lecture 5. Each slide is marked by its title. 
# Missing figures are marked with clear prompts to be implemented later.

# %%
# ## Standard Imports
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import bambi as bmb
import pandas as pd
import preliz as pz

az.style.use("arviz-docgrid")
plt.rc("figure", dpi=150)
np.random.seed(123)

# %% [markdown]
# ## Section 5A: Bayesian Workflow Recap (Figures from slides)

# %% [markdown]
# No figure generation required for these slides.

# %% [markdown]
# ## Section 5B: Why Sampling? (Figures from slides or schematic)

# %% [markdown]
# No figure generation required for these slides.

# %% [markdown]
# ## Section 5C: Grid Approximation

# %% [markdown]
# ### Slide: Visual Example: Coin Flip Grid Posterior

# %%
def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)
    likelihood = pz.Binomial(n=heads+tails, p=grid).pdf(heads)
    posterior = likelihood * prior
    posterior /= posterior.sum() * (1/grid_points)
    return grid, posterior

heads, tails = 3, 10
points = 10
grid, posterior = posterior_grid(points, heads, tails)

plt.figure()
plt.plot(grid, posterior, 'o-')
plt.title(f'Grid Posterior: Heads={heads}, Tails={tails}')
plt.xlabel('θ')
plt.yticks([])
plt.savefig('figs/grid_coin_flip.png')
plt.show()

# %% [markdown]
# ## Section 5D: Enter MCMC

# %% [markdown]
# PROMPT: Generate a schematic diagram of a simple Markov Chain process. Show a sequence of states with arrows indicating transitions and highlight the memoryless property (Slide 5D2)

# %% [markdown]
# PROMPT: Create a plot showing random sampling points scattered to approximate a target distribution (Slide 5D3)

# %% [markdown]
# PROMPT: Plot a simple trace plot showing a random walk behavior, labeled to emphasize sample dependence (Slide 5D4)

# %% [markdown]
# ## Section 5E: Metropolis Algorithm

# %% [markdown]
# ### Slide: Walkthrough: Sampling Beta(2,5)

# %%
def metropolis(dist, draws=1000):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = dist.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = dist.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

np.random.seed(3)
dist = pz.Beta(2, 5)
trace = metropolis(dist)

ax = dist.plot_pdf(legend=False)
ax.hist(trace, bins="auto", density=True, label='Estimated distribution')
ax.set_xlabel('x')
ax.set_ylabel('pdf(x)')
plt.legend()
plt.savefig('figs/metropolis_beta.png')
plt.show()

# %% [markdown]
# ### Slide: Trace Plot: Convergence Behavior

# %%
plt.figure()
plt.plot(trace, color="C0")
plt.title("Metropolis Trace Plot")
plt.xlabel("Iteration")
plt.ylabel("Sampled x")
plt.savefig('figs/metropolis_trace.png')
plt.show()

# %% [markdown]
# ## Section 5F: Diagnosing Chains (Figures mostly from slides)

# %% [markdown]
# PROMPT: Generate data from a mixture of two well-separated Gaussians, plot the histogram of the data and overlay a single Gaussian model fit to illustrate model misspecification (Slide 5F11)

# %% [markdown]
# PROMPT: Using the mixture data, fit a naive single Gaussian model using PyMC, run MCMC sampling, and plot trace plots, R-hat, and ESS diagnostics showing poor sampling behavior (Slide 5F12)

# %% [markdown]
# ## Section 5G-H-I-J (Additional Prompts)

# %% [markdown]
# PROMPT: Create a visualization of the geometric cause of divergences in HMC, showing posterior contours and trajectories getting stuck (Slide 5H2)

# %% [markdown]
# PROMPT: Visualize prior predictive mismatch: Generate prior predictive samples from an inappropriately tight prior and overlay them against typical observed data range to show mismatch (Slide 5I2)

# %% [markdown]
# PROMPT: Create a schematic of gradient steps in HMC, showing position, momentum vectors, and how gradient guides steps (Slide 5G4)

# %% [markdown]
# PROMPT: Visualize HMC trajectories: Plot a 2D posterior landscape with multiple HMC trajectories showing efficient long-distance movement (Slide 5G6)

# %% [markdown]
# PROMPT: Create a plot showing different HMC step sizes and trajectory lengths, illustrating how step size impacts sampling efficiency and divergence (Slide 5G8)

# %% [markdown]
# PROMPT: Visualize historical photo or illustration of the original Metropolis team or a placeholder image labeled 'Metropolis Algorithm History' (Slide 5E1)
