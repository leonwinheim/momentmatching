import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import mixture
import os
import pickle
import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

#******Flags******
generate_points = True
evaluate_points = False

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

#Kann ich das selbe GMM mehrmal fitten???

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

def network(z_0,layers,gm_comp, weights,single_run=False):
    """Compute stuff for the whole network"""
    # Assertions
    assert z_0.shape[1] == 1,               "Batchsize must be 1 right now"
    assert z_0.shape[2] == layers[0],       "Input size must be equal to the first layer size"
    assert len(layers) == len(gm_comp)+1,   "Number of layers must be equal to number of gm_comp + 1"
    assert len(weights) == len(layers)-1,   "Number of weights must be equal to number of layers - 1"

    # Dont save the inbetween values
    if not single_run:
        # Compute the forward pass
        z = z_0
        z_gm = z_0 
        for i in range(len(layers)-1):
            #print(f" Pass layer with {layers[i]} neurons to layer with {layers[i+1]} neurons")
            z = single_layer_pass(z, weights[i].to(device))

            z_gm = gm_fit_and_sample(z_gm,gm_comp[i],z.shape[0]) # Fit and sample the GM

            z_gm = single_layer_pass(z_gm,weights[i]) 
        
        return z,z_gm

    else:
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

    mean_rel= torch.abs(mean - gm_mean)/mean
    var_rel = torch.abs(var - gm_var)/var

    # print(f"Mean difference: {mean_diff}")
    # print(f"Variance difference: {var_diff}")

    mean_rel = mean_rel.to("cpu")
    var_rel = var_rel.to("cpu")

    return mean_rel, var_rel, mean, var

#******Parameters******
num_samples = 150000

#******Workflow******
#Generate Parameter arrays
var_range = np.arange(0.1, 2.1, 0.3)   # Variance range
width = np.arange(1, 6, 1)            # Width range
depth = np.arange(1, 6, 1)            # Depth range
components = np.arange(1, 12, 2)       # Number of components range

# Assemble parameter dict
parameter_list = []
for var in var_range:
    for w in width:
        for d in depth:
            for c in components:
                parameter_list.append({'variance': var, 'width': w, 'depth': d, 'components': c})

print(f"Number of parameter sets: {len(parameter_list)}") # Print the number of parameter sets

# Check if the pickle file exists
output_file = f"workbench/training_points_{num_samples}.pkl"
if os.path.exists(output_file):
    with open(output_file, "rb") as f:
        training_pts_in = pickle.load(f)
    print(f"Loaded training points from {output_file}")
    training_pts = training_pts_in # Initialize the list to store training points
else:
    training_pts = [] # Initialize the list to store training points
    print(f"No existing training points file found at {output_file}")

if generate_points:
    #Do the computation
    training_pts = [] # Initialize the list to store training points
    try:
        for parset in parameter_list:

            start_time = time.time() # Start the timer
            #Unpack parameters
            variance_par = parset['variance']
            w = parset['width']
            d = parset['depth'] 
            c = parset['components']

            try:
                #Checkl if a list entry in the read in traINING PTS has the first 4 entries as the current parameters already
                if any(torch.all(torch.eq(training_pt[:4], torch.tensor([variance_par,w,d,c]))) for training_pt in training_pts_in):
                    print(f"Skipping parameter set {parset} as it already exists in the training points")
                    continue
            except:
                pass

            #Generate weights
            layers = [1] + [w]*d + [1] # Add the input and output layer
            weights = generate_weights(layers,num_samples,variance_par) # Generate weights for the network

            #Generate input (standard Normal)
            z_0 = torch.randn(num_samples,1,1,device=device) # Input to the network (samples x in_neurons x batch)

            #Compute the forward pass
            res = network(z_0,layers,[c]*(len(layers)-1),weights) # Compute the forward pass

            #Evaluate moments
            mean_rel, var_rel, mean, var = evaluate_moments(res[0],res[1])

            training_pt = torch.tensor([variance_par,w,d,c,mean_rel.item(),var_rel.item(),mean.item(),var.item()])

            training_pts.append(training_pt) # Append the training point to the list

            end_time = time.time() # End the timer
            elapsed_time = end_time - start_time # Calculate the elapsed time
            print(f"Elapsed time for parameter set {parset}: {elapsed_time:.2f} seconds") # Print the elapsed time

    except:
        pass

    # Save training points to a pickle file
    output_file = f"workbench/training_points_{num_samples}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(training_pts, f)
    print(f"Training points saved to {output_file}")


if evaluate_points:
    # Structure: [variance, width, depth, components, mean_diff, var_diff, true_mean, true_var]
    data = torch.stack(training_pts_in)

    # Create a DataFrame for correlation analysis
    columns = ['Variance', 'Width', 'Depth', 'Components', 'Mean_Rel','Var_Rel', 'True_Mean', 'True_Var']
    df = pd.DataFrame(data[:,:-1].cpu().numpy(), columns=columns)

    # # Filter data for specific widths
    # widths_to_plot = [1, 3, 5]
    # plt.figure(figsize=(8, 6))

    # for width in widths_to_plot:
    #     subset = df[df['Variance'] == 0.8]
    #     subset = subset[subset['Components'] == 1]
    #     subset = subset[df['Width'] == width]
    #     subset = subset.sort_values(by='Depth', ascending=True)
    #     plt.plot(subset['Depth'], subset['Mean_Diff'], label=f'Width {width}', marker='o')

    # plt.xlabel('Depth')
    # plt.ylabel('Mean Difference')
    # plt.title('Mean Difference vs Depth for Different Widths')
    # plt.legend()
    # plt.grid(True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw={'projection': '3d'})
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Components to plot
    components_to_plot = [1, 3, 5, 7]
    variances = [0.2, 0.6, 1.0, 1.6]
    colors = ['red', 'blue', 'green', 'yellow']  # Constant colors for each variance

    # Define initial view angles for each subplot
    view_angles = [(10, 290), (10, 290), (10, 290), (10, 290)]
    z_max = 10

    for ax, specific_components, view_angle in zip(axes, components_to_plot, view_angles):
        for var, color in zip(variances, colors):
            subset = df[(df['Variance'] == var) & (df['Components'] == specific_components)]

            # Pivot the data for 3D plotting
            pivot_table = subset.pivot(index='Width', columns='Depth', values='Mean_Diff')

            # Create meshgrid for plotting
            X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
            Z = pivot_table.values

            # Plot the 3D surface
            ax.plot_surface(X, Y, Z, color=color, edgecolor='k', alpha=0.8, label=f'Variance={var}')
            ax.set_zlim(0, z_max)

        ax.set_xlabel('Depth')
        ax.set_ylabel('Width')
        ax.set_zlabel('Mean Difference')
        ax.set_title(f'Mean Difference Surface Plot\n(Components={specific_components})')
        ax.legend([f'Variance={var}' for var in variances], loc='upper left', title='Variance')

        # Set the initial view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

    plt.tight_layout()
    plt.show()


    print(f"Shape of training points: {len(training_pts_in)}")





