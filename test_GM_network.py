import numpy as np
import matplotlib.pyplot as plt
import GaussianMixtureNetwork as GMN

# Define parameters for the scalar (1D) Gaussian mixture
layers = [1,10,1]
act_func = ['relu','linear']

# Build Network
model = GMN.GaussianMixtureNetwork(layers,act_func,2,2,0.01)

model.print_network()

model.forward_moments(2.0)