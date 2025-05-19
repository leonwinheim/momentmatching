import numpy as np
import matplotlib.pyplot as plt
import gm_funcs

# Define parameters for the scalar (1D) Gaussian mixture
layers = [1,10,1]
act_func = ['relu','linear']

# Build Network
model = gm_funcs.GaussianMixtureNetwork(layers,act_func,2,2,0.01)

model.print_network()