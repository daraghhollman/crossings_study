import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import seaborn as sns

warnings.simplefilter(action="ignore", category=FutureWarning)

# Load data
combined_features = pd.read_csv("./combined_features.csv")

# Set up features and labels
x_0 = combined_features.drop(columns=["label"])  # Features
x_0 = x_0.iloc[:, 1:] # Drop the index column
y_0 = pd.Categorical(combined_features["label"]).codes # Labels

print(len(x_0), x_0.shape, y_0.shape)

_, populations = pd.factorize(combined_features["label"], sort=True)
print(populations)

coords = {"n_obs": np.arange(len(x_0)), "population": populations}

with pm.Model(coords=coords) as model_regions:
    μ = pmb.BART("μ", x_0, y_0, m=50, dims=["population", "n_obs"])
    θ = pm.Deterministic("θ", pm.math.softmax(μ, axis=0))
    y = pm.Categorical("y", p=θ.T, observed=y_0)

with model_regions:
    idata = pm.sample(chains=4, compute_convergence_checks=False)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)


pmb.plot_variable_importance(idata, μ, x_0, method="VI");
plt.show()
