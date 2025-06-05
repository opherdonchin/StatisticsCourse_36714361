#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[2]:


az.style.use("arviz-whitegrid")
plt.rc('figure', dpi=450)


# In[3]:


np.random.seed(123)
trials = 4
theta_real = 0.35 # unknown value in a real experiment
data = pz.Binomial(n=1, p=theta_real).rvs(trials)
data


# In[4]:


with pm.Model() as our_first_model:
    theta = pm.Beta('theta', alpha=1., beta=1.)
    y = pm.Bernoulli('y', p=theta, observed=data)


# In[5]:


idata = pm.sample(1000, model=our_first_model, random_seed=4591)


# In[6]:


idata


# In[27]:


pm.model_to_graphviz(our_first_model)


# In[28]:


our_first_model.basic_RVs


# In[11]:


pytensor.dprint(y)


# In[12]:


# Generate the graph image data without writing to disk
img_data = pydotprint(y, format="png", var_with_name_simple=True, return_image=True)

# Display the image in a Jupyter/Colab notebook
from IPython.display import Image, display
display(Image(img_data))


# In[37]:


for i in range(10):
    print(f"Pytensor sample {i}: {theta.eval()}")


# In[33]:


for i in range(10):
    print(f"Sample {i}: {pm.draw(theta)}")


# In[39]:


for i in range(10):
    random_theta, random_y = pm.draw([theta, y])
    print(f"Sample {i}: theta = {random_theta}, y = {random_y}")


# In[47]:


from scipy import special 

sample_theta  = pm.draw(theta)    

logp_func = our_first_model.compile_logp()
joint_logp = logp_func({"theta_logodds__": special.logit(sample_theta)})

print("Sampled theta:", sample_theta)
print("Sampled y:", data)
print("Joint log probability:", joint_logp)

