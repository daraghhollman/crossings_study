"""
Loads data and creates the model
"""
import datetime as dt

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import spiceypy as spice

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag

RANDOM_SEED = 8457

populations = pd.read_csv("./region_data_5_mins.csv")

y_0 = pd.Categorical(populations["region"]).codes
x_0 = populations[["mag_total", "mag_x", "mag_y", "mag_z", "mag_variability"]]

_, regions = pd.factorize(populations["region"], sort=True)

coords = {"n_obs": np.arange(len(x_0)), "regions": regions}

with pm.Model(coords=coords) as regions_model:
    μ = pmb.BART("μ", x_0, y_0, m=50, dims=["regions", "n_obs"])
    θ = pm.Deterministic("θ", pm.math.softmax(μ, axis=0))
    y = pm.Categorical("y", p=θ.T, observed=y_0)

with regions_model:
    idata = pm.sample(chains=4, compute_convergence_checks=False, random_seed=123)
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)


pmb.plot_variable_importance(idata, μ, x_0, method="VI", random_seed=RANDOM_SEED)
plt.show()

ax = az.plot_ppc(idata, kind="kde", num_pp_samples=200, random_seed=123)
# plot aesthetics
ax.set_ylim(0, 0.7)
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_ylabel("Probability")
ax.set_xlabel("Region")

plt.show()


philpott_crossings = boundaries.Load_Crossings("/home/daraghhollman/Main/mercury/philpott_2020_reformatted.csv")

root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
metakernel = "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
spice.furnsh(metakernel)

start_time = dt.datetime(year=2013, month=6, day=1, hour=9, minute=55)
end_time = dt.datetime(year=2013, month=6, day=1, hour=10, minute=10)

prediction_data = pd.DataFrame({"input": })
