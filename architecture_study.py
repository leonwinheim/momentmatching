import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # Limit the number of threads used by joblib, windows 11 stuff

#ALL COMPUTATIONS HERE ARE WITHOUT BIAS

#******Define Functions******
def act(x):
    """Return Relu or Leaky Relu"""
    leaky = False
    leaky_slope = 0.05
    if leaky:
        x = torch.where(x > 0, x, x * leaky_slope)
    else:
        x = torch.where(x > 0, x, 0)
    return x

def generate_weights(layers,num_samples,variance_par):
    """Generates a list of random weight matrices for the network"""
    weights = []
    for i in range(len(layers)-1):
        #******Set Moments of weights******
        mean = torch.zeros(layers[i+1]*layers[i])               # Shape: (layers[i+1] * layers[i],)
        cov = variance_par*torch.eye(layers[i+1]*layers[i])     # Shape: (layers[i+1] * layers[i], layers[i+1] * layers[i])

        # Generate random weights from a multivariate normal distribution
        mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        w = mvn.sample((num_samples,)) 
        w = w.view(num_samples, layers[i+1], layers[i])         # Reshape to (samples x out_neurons x in_neurons)
        weights.append(w)                                       # Append to weights list

    return weights

def single_layer_pass(z:torch.tensor,w:torch.tensor):
    """z is (samples x in_neurons x batch)
        w is (samples x out_neurons x in_neurons )
        a is (samples x out_neurons x batch)
        z is like a but activated"""
    a = torch.bmm(z,w.transpose(1,2))   # (samples x out_neurons x batch)
    return act(a)                       # (samples x out_neurons x batch)

def network(z_0,layers,gm_comp, weights):
    """Compute stuff for the whole network"""
    # Assertions
    assert z_0.shape[1] == 1,               "Batchsize must be 1 right now"
    assert z_0.shape[2] == layers[0],       "Input size must be equal to the first layer size"
    assert len(layers) == len(gm_comp)+1,   "Number of layers must be equal to number of gm_comp + 1"
    assert len(weights) == len(layers)-1,   "Number of weights must be equal to number of layers - 1"

    # Compute the forward pass
    z = z_0
    z_gm = z_0 
    for i in range(len(layers)-1):
        #print(f" Pass layer with {layers[i]} neurons to layer with {layers[i+1]} neurons")
        z = single_layer_pass(z,weights[i])
        z_gm = single_layer_pass(z_gm,weights[i]) 
    
    return z,z_gm

def gm_fit_and_sample(z,components,num_samples):
    """Take a sample based distribution and fit a GM to it.
        Then, sample and return the intermediate samples"""
    z_out = torch.zeros_like(z) # Initialize the output tensor
    gm = mixture.GaussianMixture(n_components=components, covariance_type="full")
    for i in range(z.shape[2]):
        gm.fit(z[:,0,i].reshape(-1,1)) # Fit the GM to the data
        z_out[:,0,i] = torch.tensor(gm.sample(num_samples)[0], dtype=torch.float32).squeeze()# Sample from the GM
    
    return z_out

def evaluate_moments(z,z_gm):
    """Evalute the moments of the distribution and compare for one run"""
    # Compute the mean and variance of the distribution
    mean = torch.mean(z, dim=0)
    var = torch.var(z, dim=0)

    # Compute the mean and variance of the Gaussian Mixture
    gm_mean = torch.mean(z_gm, dim=0)
    gm_var = torch.var(z_gm, dim=0)

    # Print the results
    print("Mean of direct version: ", mean)
    print("Variance of direct version: ", var)
    print("Mean of Gaussian Mixture version: ", gm_mean)
    print("Variance of Gaussian Mixture version: ", gm_var)

    mean_diff = torch.abs(mean - gm_mean)
    var_diff = torch.abs(var - gm_var)

    return mean_diff, var_diff

#******Parameters******
num_samples = 100000

layers_ex = [1,3,2,1]   # First entry is input size, has no weight
gm_ex = [2,2,2] 	    # Number of Gaussians between each layer (len(gm_ex) = len(layers_ex)-1)
variance_par = 1        # Variance of the Gaussian distributions
weights_ex = generate_weights(layers_ex,num_samples,variance_par) # Generate weights for the network

z_0 = torch.randn(num_samples,1,1) # Input to the network (samples x in_neurons x batch)

fit = gm_fit_and_sample(z_0,gm_ex[0],num_samples) # Fit a GM to the input data
print(f"Shape of fit: {fit.shape}") # Check the shape of the output


res = network(z_0,layers_ex,gm_ex,weights_ex) # Compute the forward pass

#******Workflow******
#Generate Parameter arrays
#Compute Labels (mean error ansd cov error)

#save them
#train network/frit function





