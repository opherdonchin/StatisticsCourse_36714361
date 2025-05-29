#!/usr/bin/env python
# coding: utf-8

# In[2]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz


# In[3]:


az.style.use("arviz-whitegrid")
plt.rc('figure', dpi=450)


# In[4]:


data = np.loadtxt("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/chemical_shifts.csv")


# In[5]:


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


# In[6]:


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


# In[7]:


mu = [0, 0, 2]
sigma = [1, 0.5, 1]
_, ax = plt.subplots(figsize=(6, 3))
for i in range(3):
    normal_distribution = pz.Normal(mu=mu[i], sigma=sigma[i])
    normal_distribution.plot_pdf(ax=ax)

plt.show()


# In[8]:


l = [0.0, 49.0, 55.0]
h = [100.0, 70.0, 65.0]
_, ax = plt.subplots(figsize=(6, 3))
for i in range(3):
    uniform_dist = pz.Uniform(lower=l[i], upper=h[i])
    uniform_dist.plot_pdf(ax=ax)
    
plt.show()


# In[9]:


sigma = [2.0, 5.0, 10.0]

_, ax = plt.subplots(figsize=(6, 3))
for i in range(3):
    halfnormal_dist = pz.HalfNormal(sigma=sigma[i])
    halfnormal_dist.plot_pdf(ax=ax)
    
plt.show()


# In[10]:


l_mu = 40
h_mu = 70
σ_σ = 5


# In[11]:


with pm.Model() as model_g:
    μ = pm.Uniform('μ', lower=l_mu, upper=h_mu)
    σ = pm.HalfNormal('σ', sigma=σ_σ)
    y = pm.Normal('y', mu=μ, sigma=σ, observed=data)


# In[12]:


# Sample from the prior distribution
with model_g:
    idata_g = pm.sample_prior_predictive(samples=1000, random_seed=42)
idata_g


# In[13]:


plt.figure(figsize=(12, 6))
ax = az.plot_ppc(idata_g, group='prior', num_pp_samples=1, mean=False, observed=True, random_seed=14)
ax.get_lines()[2].set_alpha(1.0)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Prior Predictive from model_g')

plt.legend()
plt.show()


# In[14]:


plt.figure(figsize=(10, 6))
ax = az.plot_ppc(idata_g, group='prior', num_pp_samples=4, mean=False, observed=True, random_seed=14)

for l in ax.get_lines()[2:]:
    l.set_alpha(0.9)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Prior Predictive from model_g')

plt.legend()
plt.show()


# In[15]:


plt.figure(figsize=(10, 6))
ax = az.plot_ppc(idata_g, group='prior', num_pp_samples=20, mean=False, observed=True, random_seed=14)

for l in ax.get_lines()[2:]:
    l.set_alpha(0.9)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Prior Predictive from model_g')

plt.legend()
plt.show()


# In[16]:


with model_g:
    idata_g.extend(pm.sample(random_seed=4591), join='right')


# In[17]:


idata_g


# In[18]:


az.plot_trace(idata_g, compact=False);


# In[19]:


az.plot_pair(idata_g, kind='scatter', marginals=True);


# In[20]:


az.plot_pair(idata_g, kind='kde', marginals=True);


# In[21]:


az.summary(idata_g, kind='stats', round_to=2)


# In[22]:


pm.sample_posterior_predictive(idata_g, model=model_g, extend_inferencedata=True, random_seed=4591)


# In[23]:


az.plot_ppc(idata_g, num_pp_samples=100, figsize=(10, 4), colors=["C1", "C0", "C1"])


# In[24]:


for nu in [1, 2, 10]:
    pz.StudentT(nu, 0, 1).plot_pdf(support=(-5, 5), figsize=(12, 4))

ax = pz.StudentT(np.inf, 0, 1).plot_pdf(support=(-5, 5), figsize=(12, 4), color="k")
ax.get_lines()[-1].set_linestyle("--")
pz.internal.plot_helper.side_legend(ax)


# In[25]:


for nu in [1, 2, 10]:
    pz.StudentT(nu, 0, 1).plot_pdf(support=(-7, 7), figsize=(12, 4))

ax = pz.StudentT(np.inf, 0, 1).plot_pdf(support=(-7, 7), figsize=(12, 4), color="k")
ax.get_lines()[-1].set_linestyle("--")
pz.internal.plot_helper.side_legend(ax)
ax.set_ylim(0, 0.07)
plt.show()


# In[26]:


l_mu = 40
h_mu = 70
σ_σ = 10
λ_ν = 1/30


# In[27]:


with pm.Model() as model_t:
    μ = pm.Uniform('μ', l_mu, h_mu)
    σ = pm.HalfNormal('σ', sigma=σ_σ)
    ν = pm.Exponential('ν', λ_ν)
    y = pm.StudentT('y', nu=ν, mu=μ, sigma=σ, observed=data)


# In[28]:


idata_t = pm.sample(random_seed=4591, model=model_t)


# In[29]:


az.plot_trace(idata_t, compact=False);


# In[30]:


az.summary(idata_t, kind="stats", round_to=2)


# In[31]:


ax


# In[32]:


ax = az.plot_pair(idata_t, kind='kde', marginals=True)
for row in ax:
	for subplot in row:
		if subplot is not None:
			subplot.set_xlabel(subplot.get_xlabel(), fontsize=30)
			subplot.set_ylabel(subplot.get_ylabel(), fontsize=30)
			subplot.tick_params(axis='both', which='major', labelsize=26)


# In[33]:


ax = az.plot_pair(idata_t, var_names=['σ', 'ν'], kind='kde', marginals=False)

ax.set_xlabel(ax.get_xlabel(), fontsize=18)
ax.set_ylabel(ax.get_ylabel(), fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlim(1.2, 3.0)
ax.set_ylim(0, 15)


# In[34]:


pm.sample_posterior_predictive(idata_t, model=model_t, extend_inferencedata=True, random_seed=4591)


# In[35]:


ax = az.plot_ppc(idata_t, num_pp_samples=100, figsize=(10, 4), colors=["C1", "C0", "C1"])
ax.set_xlim(39, 70)


# In[36]:


tips = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data//tips.csv")
tips.tail()


# In[40]:


_, ax = plt.subplots(figsize=(10, 4))
az.plot_forest(tips.pivot(columns="day", values="tip").to_dict("list"),
               kind="ridgeplot",
               hdi_prob=1,
               ax=ax)
ax.set_xlabel("Tip amount")
ax.set_ylabel("Day")

