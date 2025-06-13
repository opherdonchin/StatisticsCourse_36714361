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
import bambi as bmb
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz
import xarray as xr

from scipy import stats

# %%
az.style.use("arviz-whitegrid")
plt.rc('figure', dpi=450)


# %%
def posterior_grid(grid_points=50, heads=6, tails=9):
    """ 
    A grid implementation for the coin-flipping problem 
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = pz.Binomial(n=heads+tails, p=grid).pdf(heads)
    posterior = likelihood * prior
    posterior /= posterior.sum() * (1/grid_points)
    return grid, posterior


# %%
bernoulli_data = np.repeat([0, 1], (10, 3))
points = 10
h = bernoulli_data.sum()
t = len(bernoulli_data) - h
grid, posterior = posterior_grid(points, h, t) 

plt.plot(grid, posterior, 'o-')

plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ');

# %%
# Define parameters for Gaussian prior
prior_mean = 0
prior_var = 2  # prior variance
prior_std = np.sqrt(prior_var)

# Create some synthetic data
n_data = 10
true_mean = 2
data_std = 1
normal_data = np.random.normal(true_mean, data_std, n_data)

# Calculate posterior parameters (conjugate update for Gaussian)
# For a Gaussian likelihood with known variance:
# posterior_precision = prior_precision + n * data_precision
# posterior_mean = (prior_mean * prior_precision + sum(x_i) * data_precision) / posterior_precision
data_var = data_std**2
posterior_var = 1 / (1/prior_var + n_data/data_var)
posterior_std = np.sqrt(posterior_var)
posterior_mean = posterior_var * (prior_mean/prior_var + np.sum(normal_data)/data_var)

# Define x values for plotting
x = np.linspace(posterior_mean - 4*posterior_std, posterior_mean + 4*posterior_std, 1000)

# Sample from the posterior
n_samples = 50
posterior_samples = np.random.normal(posterior_mean, posterior_std, n_samples)

# Plot the distributions and samples
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, stats.norm.pdf(x, posterior_mean, posterior_std), 'g-', label='True Posterior')
# Plot vertical lines for the posterior samples
ax.vlines(posterior_samples, ymin=0, ymax=stats.norm.pdf(posterior_samples, posterior_mean, posterior_std), alpha=0.1, color='b')

ax.set_xlabel('θ')
ax.set_ylabel('Density')
ax.legend()

# %%
# Define grid of evenly spaced points around the posterior mean
x_grid = np.linspace(posterior_mean - 4*posterior_std, 
                      posterior_mean + 4*posterior_std, 1000)

# Calculate posterior density at each point
posterior_density = stats.norm.pdf(x_grid, posterior_mean, posterior_std)

# Create 100 evenly spaced samples for vertical lines
n_samples = 100
grid_samples = np.linspace(posterior_mean - 3*posterior_std, 
                           posterior_mean + 3*posterior_std, n_samples)

# Plot the distributions and evenly spaced samples
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_grid, posterior_density, 'g-', label='True Posterior', lw=2)

# Plot vertical lines for the evenly spaced samples
ax.vlines(grid_samples, 
          ymin=0, 
          ymax=stats.norm.pdf(grid_samples, posterior_mean, posterior_std), 
          alpha=0.2, color='b')

ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Distribution with Grid Samples', fontsize=14)
ax.legend()

plt.show()

# %%
bernoulli_data = pd.DataFrame(bernoulli_data, columns=["w"])
priors = {"Intercept": bmb.Prior("Uniform", lower=0, upper=1)}
model = bmb.Model("w ~ 1", data=bernoulli_data, family="bernoulli", priors=priors, link="identity")
results = model.fit(draws=4000, inference_method="laplace")

# %%
_, ax = plt.subplots(figsize=(12, 5))
# analytical calculation
x = np.linspace(0, 1, 100)
plt.plot(x, pz.Beta(h+1, t+1).pdf(x), label='True posterior', lw=3)

# quadratic approximation
az.plot_kde(results.posterior["Intercept"].values, label='Quadratic approximation',
            ax=ax, plot_kwargs={"color": "C1", "lw": 3})

ax.set_title(f'heads = {h}, tails = {t}')
ax.set_xlabel('θ')
ax.set_yticks([])

# %%
# Generate samples from the quadratic approximation (Laplace/Normal approximation)
n_samples = 100
laplace_samples = np.random.normal(results.posterior["Intercept"].mean().values, 
                                   results.posterior["Intercept"].std().values,
                                   n_samples)

# Ensure samples are within the bounds of [0, 1] for the Bernoulli parameter
laplace_samples = np.clip(laplace_samples, 0, 1)

# Plot the analytic Beta posterior and the samples from the Laplace approximation
fig, ax = plt.subplots(figsize=(12, 5))

# Plot true posterior (Beta distribution)
x = np.linspace(0, 1, 100)
ax.plot(x, pz.Beta(h+1, t+1).pdf(x), label='True posterior', lw=3)

# Plot Laplace approximation curve
az.plot_kde(results.posterior["Intercept"].values, label='Quadratic approximation',
            ax=ax, plot_kwargs={"color": "C1", "lw": 3})

# Plot vertical lines for samples from the Laplace approximation
# but with heights from the true posterior (Beta distribution)
for sample in laplace_samples:
    # Calculate height using the true posterior (Beta) density
    height = pz.Beta(h+1, t+1).pdf(sample)
    ax.vlines(sample, 0, height, alpha=0.2, color='C1')

ax.set_title(f'heads = {h}, tails = {t}')
ax.set_xlabel('θ')
ax.set_yticks([])
ax.legend()

# %%
# Generate three different Monte Carlo simulations with different N values
N_values = [1000, 10000,100000]

for N in N_values:
    # Run the Monte Carlo simulation for approximating π
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / np.pi) * 100
    
    outside = np.invert(inside)
    
    # Create a separate figure for each N value
    _, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x[inside], y[inside], 'C2.')
    ax.plot(x[outside], y[outside], 'C1.')
    ax.axis('square')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'N = {N}\nπ* = {pi:4.3f}\nerror = {error:.3f}%', fontsize=18)
    plt.show()  # Display each figure individually


# %%
x = np.arange(1, 8)
p = x / np.sum(x)  # normalize probabilities

plt.figure(figsize=(7, 2))
plt.bar(x, p, color='C0')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('p(x) = x for x = 1..7')
plt.xticks(x)
plt.ylim(0, max(p) * 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

states = np.arange(1, 8)
p = states / np.sum(states)

chain = [4]
num_steps = 500

for _ in range(num_steps):
    current = chain[-1]
    proposal = current + np.random.choice([-1, 1])
    if proposal < 1 or proposal > 7:
        proposal = current
    accept_prob = min(1, p[proposal - 1] / p[current - 1])
    if np.random.rand() < accept_prob:
        chain.append(proposal)
    else:
        chain.append(current)

# Plotting
plt.figure(figsize=(7, 4))
# Plot step number nonlinearly scaled
y_positions = np.log1p(np.arange(1, num_steps+2))  # log scale for step number
plt.plot(chain, y_positions, color='b', marker='o') # drawstyle='steps-post', 

plt.xlabel('State')
plt.ylabel('Step')
plt.title('MCMC Chain Trace (Nonlinear Y-axis)')
plt.yticks(
    np.log1p([1, 2, 5, 10, 100, 500]),
    ['1', '2', '5', '10', '100', '500']
)
# plt.gca().invert_yaxis()  # Step 1 at the bottom, Step 500 at the top
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# %%
chain = [4]
num_steps = 10000

for _ in range(num_steps):
    current = chain[-1]
    proposal = current + np.random.choice([-1, 1])
    if proposal < 1 or proposal > 7:
        proposal = current
    accept_prob = min(1, p[proposal - 1] / p[current - 1])
    if np.random.rand() < accept_prob:
        chain.append(proposal)

# Generate samples from the chain
samples = np.array(chain)

# Create the figure
plt.figure(figsize=(7, 2))
plt.hist(samples, bins=range(1, 9), density=True, color='C0')
plt.xlabel('State')
plt.ylabel('Density')
plt.title('Distribution of MCMC Samples')
plt.xticks(range(1, 8))
plt.ylim(0, 0.4)  # Setting similar y-axis limits as the original distribution
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# States 1 to 7
states = np.arange(1, 8)

# Transition matrix
transition = np.zeros((7, 7))
for i in range(7):
    for j in [-1, 1]:  # Propose left or right
        neighbor = i + j
        if 0 <= neighbor < 7:
            # Acceptance probability = min(1, p(neighbor)/p(current)) = neighbor+1/(i+1)
            accept_prob = min(1, (neighbor + 1) / (i + 1))
            transition[i, neighbor] += 0.5 * accept_prob
            transition[i, i] += 0.5 * (1 - accept_prob)  # Stay if rejected
        else:
            transition[i, i] += 0.5  # Boundary reflects, stay put

# Normalize rows (not strictly necessary, but double-checking)
transition = transition / transition.sum(axis=1, keepdims=True)

# Initial distribution: all probability at state 4 (index 3)
initial = np.zeros(7)
initial[2] = 1.0

# Time steps to plot
time_steps = [1, 2, 3, 4, 8, 12]
distributions = []

current = initial.copy()
for t in range(1, max(time_steps)+1):
    current = current @ transition
    if t in time_steps:
        distributions.append(current.copy())

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)

for ax, dist, t in zip(axes.flatten(), distributions, time_steps):
    ax.bar(states, dist, color='C0')
    ax.set_title(f'Time Step {t}')
    ax.set_xticks(states)
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.suptitle('Development of Probability Over Time', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%
# Continue from previous time steps analysis
# Calculate distributions for time steps 13, 14, 99, and the target (stationary) distribution
time_steps_new = [13, 14, 99]
distributions_new = []

# Start from the last distribution we already have (t=12)
current = distributions[-1].copy()
for t in range(13, max(time_steps_new)+1):
    current = current @ transition
    if t in time_steps_new:
        distributions_new.append(current.copy())

# Calculate the stationary/target distribution
target_dist = states / np.sum(states)  # This is the target distribution p(x) = x/sum(x)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
axes = axes.flatten()

# Plot the time steps
for ax, dist, t in zip(axes[:3], distributions_new, time_steps_new):
    ax.bar(states, dist, color='C0')
    ax.set_title(f'Time Step {t}')
    ax.set_xticks(states)
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Plot the target distribution
axes[3].bar(states, target_dist, color='C1')
axes[3].set_title('Target Distribution')
axes[3].set_xticks(states)
axes[3].set_ylim(0, 0.4)
axes[3].grid(axis='y', linestyle='--', alpha=0.7)

fig.suptitle('Development of Probability Over Time', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
coords = {"data": np.arange(10)}
with pm.Model(coords=coords) as model_cm:
    a = pm.HalfNormal('a', 10)
    b = pm.Normal('b', 0, a, dims='data')
    idata_cm = pm.sample(random_seed=367)

# %%

with pm.Model(coords=coords) as model_ncm:
    a = pm.HalfNormal('a', 10)

    b_offset = pm.Normal('b_offset', mu=0, sigma=1, dims='data')
    b = pm.Deterministic('b', 0 + b_offset * a, dims='data')
    idata_ncm = pm.sample(random_seed=367)

# %%
ax = az.plot_trace(idata_cm, var_names=['a', "b"], coords={'data': [0]}, divergences='top', chain_prop={"ls": "-"}, compact=False)
ax[0, 0].get_figure().suptitle('Centered model', fontsize=20)

# %%
ax = az.plot_trace(idata_ncm, var_names=['a', "b"], divergences='top', coords={'data': [0]}, chain_prop={"ls": "-"}, compact=False)
ax[0, 0].get_figure().suptitle('Non-centered model', fontsize=20)

# %%
rhat.x

# %%
# Parameters
num_draws = 10000
plateau_length = 6000
chain_means_start = [-6, -2, 2, 6]
num_chains = len(chain_means_start)

# Generate chains
chains = []
for start_mean in chain_means_start:
    sample_nums = np.arange(1, num_draws + 1)
    dynamic_mean = start_mean * np.maximum(0, (1 - sample_nums / (num_draws - plateau_length)))
    chain_samples = np.random.normal(loc=dynamic_mean, scale=1.0, size=num_draws)
    chains.append(chain_samples)

chains = np.array(chains)  # shape: (num_chains, num_draws)
idata = az.convert_to_inference_data(chains)

# --- Calculate incremental R-hat first ---
step = 50
rhat_values = []
iterations = []
for i in range(step, num_draws + 1, step):
    start_slice = np.maximum(0, i - plateau_length)
    partial = idata.sel(draw=slice(start_slice, i))
    rhat = az.rhat(partial)
    rhat_mean = rhat.x.values
    rhat_values.append(rhat_mean)
    iterations.append(i)

# Determine R-hat range for scaling
rhat_min = min(rhat_values)
rhat_max = max(rhat_values)

# Scale chains into R-hat range
scaled_chains = np.interp(chains, (chains.min(), chains.max()), (rhat_min, rhat_max))

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot scaled chains
for c in range(num_chains):
    plt.plot(np.arange(1, num_draws+1), scaled_chains[c], label=f'Chain {c+1}', alpha=0.8)

# Plot actual R-hat (native units)
plt.plot(iterations, rhat_values, color='black', linewidth=2, label='R-hat')

plt.xlabel("Iteration")
plt.ylabel("R-hat & Scaled Chains")
plt.title(f"R-hat calculated on a window of {plateau_length} samples")
plt.legend()
plt.grid(alpha=0.6, linestyle='--')
plt.tight_layout()
plt.show()



# %%
rhat.x.values

# %%
num_draws = idata_cm.posterior.sizes['draw']

step = 10  # Compute every 100 draws (adjustable)
rhat_values = []
iterations = []

# Loop through increasing number of draws
for i in range(step, num_draws + 1, step):
    partial = idata_cm.sel(draw=slice(0, i))  # Select first i draws
    rhat = az.rhat(partial)
    # Option: Average R-hat over all variables
    rhat_mean = rhat.b[0].mean().values
    rhat_values.append(rhat_mean)
    iterations.append(i)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(iterations, rhat_values)
plt.xlabel("Num iterations")
plt.ylabel("Shrink factor (R-hat)")
plt.title("Centered model")
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.95, 1.25)
plt.show()

# %%
num_draws = idata_ncm.posterior.sizes['draw']

step = 10  # Compute every 100 draws (adjustable)
rhat_values = []
iterations = []

# Loop through increasing number of draws
for i in range(step, num_draws + 1, step):
    partial = idata_ncm.sel(draw=slice(0, i))  # Select first i draws
    rhat = az.rhat(partial)
    # Option: Average R-hat over all variables
    rhat_mean = rhat.b[0].mean().values
    rhat_values.append(rhat_mean)
    iterations.append(i)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(iterations, rhat_values)
plt.xlabel("Num iterations")
plt.ylabel("Shrink factor (R-hat)")
plt.title("Non-centered model")
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.95, 1.25)
plt.show()

# %%
rhat = az.rhat(idata_cm, var_names=['a', 'b'])
rhat.b[0].values

# %%
az.plot_autocorr(idata_cm, var_names=['a'], combined=True, figsize=(6, 3))
plt.title('Centered model')
plt.xlabel('Lag ($\\delta$)')
plt.ylabel('Autocorrelation ($\\rho$)')

# %%
az.plot_autocorr(idata_ncm, var_names=['a'], combined=True, figsize=(6, 3))
plt.title('Non-Centered model')
plt.xlabel('Lag ($\\delta$)')
plt.ylabel('Autocorrelation ($\\rho$)')

# %%
az.summary(idata_cm, coords={'data': [0]}, kind='diagnostics')

# %%
az.summary(idata_ncm, coords={'data': [0]}, kind='diagnostics')


# %%
def rad_dist(Y):
    return np.sqrt(np.sum(Y**2))


fig, ax = plt.subplots(1, 1, figsize=[7, 3])
xvar = np.linspace(0, 36, 200)
# the book code is wrapped in a loop to reproduce Figure 9.4
for D in [1, 10, 100, 1000]:
    T = int(1e3)
    Y = stats.multivariate_normal(np.zeros(D), np.identity(D)).rvs(T)

    Rd = list(map(rad_dist, Y))

    kde = stats.gaussian_kde(Rd)
    yvar = kde(xvar)

    ax.plot(xvar, yvar, color="k")
    ax.text(np.mean(Rd), np.max(yvar) * 1.02, f"D: {D}")
    ax.set_xlabel('Distance from mode')
    ax.set_ylabel('Density')
    ax.set_yticks([])


# %%
idata_cm.posterior["log(a)"] = np.log(idata_cm.posterior["a"])

_, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 4))

az.plot_pair(
    idata_cm,
    var_names=["b", "log(a)"],
    coords={"data": [0]},
    divergences=True,
    scatter_kwargs={"color": "cyan"},
    divergences_kwargs={"color": "k", "marker": ".", "mec": None},
    ax=axes[0],
)
axes[0].set_title("model_c")

idata_ncm.posterior["log(a)"] = np.log(idata_ncm.posterior["a"])

az.plot_pair(
    idata_ncm,
    var_names=["b", "log(a)"],
    coords={"data": [0]},
    divergences=True,
    scatter_kwargs={"color": "cyan"},
    divergences_kwargs={"color": "k", "marker": ".", "mec": None},
    ax=axes[1],
)
axes[1].set_title("model_nc")

az.plot_pair(
    idata_ncm,
    var_names=["b_offset", "log(a)"],
    coords={"data": [0]},
    divergences=True,
    scatter_kwargs={"color": "cyan"},
    divergences_kwargs={"color": "k", "marker": ".", "mec": None},
    ax=axes[2],
)
axes[2].set_title("model_nc")
