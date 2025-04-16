import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture
import os
import pickle
import time

os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # Limit the number of threads used by joblib, windows 11 stuff

# Checkl for CUDA Availability
try_cuda = False

if torch.cuda.is_available() and try_cuda:
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

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
        z = single_layer_pass(z, weights[i].to(device))

        z_gm = gm_fit_and_sample(z_gm,gm_comp[i],z.shape[0]) # Fit and sample the GM

        z_gm = single_layer_pass(z_gm,weights[i]) 
    
    return z,z_gm

def gm_fit_and_sample(z,components,num_samples):
    """Take a sample based distribution and fit a GM to it.
        Then, sample and return the intermediate samples"""
    z_out = torch.zeros_like(z).to(device) # Initialize the output tensor
    gm = mixture.GaussianMixture(n_components=components, covariance_type="full")
    for i in range(z.shape[2]):
        gm.fit(z[:,0,i].reshape(-1,1)) # Fit the GM to the data
        z_out[:,0,i] = torch.tensor(gm.sample(num_samples)[0], dtype=torch.float32,device=device).squeeze()# Sample from the GM
    
    return z_out

def evaluate_moments(z,z_gm):
    """Evalute the moments of the distribution and compare for one run"""
    # Compute the mean and variance of the distribution
    mean = torch.mean(z, dim=0)
    var = torch.var(z, dim=0)

    # Compute the mean and variance of the Gaussian Mixture
    gm_mean = torch.mean(z_gm, dim=0)
    gm_var = torch.var(z_gm, dim=0)

    mean_diff = torch.abs(mean - gm_mean)
    var_diff = torch.abs(var - gm_var)

    # print(f"Mean difference: {mean_diff}")
    # print(f"Variance difference: {var_diff}")

    mean_diff = mean_diff.to("cpu")
    var_diff = var_diff.to("cpu")

    return mean_diff, var_diff

#******Parameters******
num_samples = 100000

#******Workflow******
#Generate Parameter arrays
var_range = np.arange(0.1, 2.1, 0.1)   # Variance range
width = np.arange(1, 6, 2)            # Width range
depth = np.arange(1, 6, 2)            # Depth range
components = np.arange(1, 11, 1)       # Number of components range

# Assemble parameter dict
parameter_list = []
for var in var_range:
    for w in width:
        for d in depth:
            for c in components:
                parameter_list.append({'variance': var, 'width': w, 'depth': d, 'components': c})

print(f"Number of parameter sets: {len(parameter_list)}") # Print the number of parameter sets
exit()

#Do the computation
training_pts = [] # Initialize the list to store training points
for parset in parameter_list:
    start_time = time.time() # Start the timer
    #Unpack parameters
    variance_par = parset['variance']
    w = parset['width']
    d = parset['depth'] 
    c = parset['components']

    #Generate weights
    layers = [1] + [w]*d + [1] # Add the input and output layer
    weights = generate_weights(layers,num_samples,variance_par) # Generate weights for the network

    print(layers)

    #Generate input (standard Normal)
    z_0 = torch.randn(num_samples,1,1,device=device) # Input to the network (samples x in_neurons x batch)

    #Compute the forward pass
    res = network(z_0,layers,[c]*(len(layers)-1),weights) # Compute the forward pass

    #Evaluate moments
    mean_diff, var_diff = evaluate_moments(res[0],res[1])

    training_pt = torch.tensor([variance_par,w,d,c,mean_diff.item(),var_diff.item()])

    training_pts.append(training_pt) # Append the training point to the list

    end_time = time.time() # End the timer
    elapsed_time = end_time - start_time # Calculate the elapsed time
    print(f"Elapsed time for parameter set {parset}: {elapsed_time:.2f} seconds") # Print the elapsed time

# Save training points to a pickle file
output_file = "workbench/training_points.pkl"
with open(output_file, "wb") as f:
    pickle.dump(training_pts, f)
print(f"Training points saved to {output_file}")








