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
generate_points = False
evaluate_points = True
single_run = False

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

#Kann ich das selbe GMM mehrmal fitten??? Ja kann ich

#******Define Functions******
def act(x):
    """Return Relu or Leaky Relu"""
    leaky = True
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
        #Save intermediate values
        z_hist = []
        z_gm_hist = []

        # Compute the forward pass
        z = z_0
        z_gm = z_0

        z_hist.append(z)
        z_gm_hist.append(z_gm)
        

        for i in range(len(layers)-1):
            #print(f" Pass layer with {layers[i]} neurons to layer with {layers[i+1]} neurons")
            z = single_layer_pass(z, weights[i].to(device))
            z_hist.append(z)

            z_gm = single_layer_pass(z_gm,weights[i]) 

            z_gm = gm_fit_and_sample(z_gm,gm_comp[i],z.shape[0]) # Fit and sample the GM
            z_gm_hist.append(z_gm)
        
        return z,z_gm,z_hist,z_gm_hist

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
if generate_points or evaluate_points:
    #Generate Parameter arrays
    var_range = np.array([1.0])     # Variance range
    width = np.array([1, 2, 5, 10, 20, 30])                     # Width range
    depth = np.array([1, 3, 5, 10])                     # Depth range
    #components = np.array([1, 3, 5, 7, 9])          # Number of components range
    components = np.array([1,2,5,10,15])          # Number of components range

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
    #training_pts = [] # Initialize the list to store training points
    try:
        for parset in parameter_list:
            #Reset seed for reproducibility
            torch.manual_seed(0) 

            # Start the timer
            start_time = time.time()

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
    columns = ['Variance', 'Width', 'Depth', 'Components', 'Mean_Rel','Var_Rel', 'True_mean', 'True_var']
    df = pd.DataFrame(data.cpu().numpy(), columns=columns)

    # # Print unique variances
    # unique_variances = df['Variance'].unique()
    # print("Unique Variances:", unique_variances)

    # #******Do the Mean Surface plot******
    # fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw={'projection': '3d'})
    # axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # # Variances to plot
    # variances_to_plot = [0.1, 0.5, 1.0, 1.5]
    # components = [1, 3, 5, 9, 15]
    # colors = ['red', 'blue', 'green', 'yellow','purple']  # Constant colors for each component

    # # Define initial view angles for each subplot
    # angle_a = 220
    # angle_b = 5
    # view_angles = [(angle_b, angle_a), (angle_b, angle_a), (angle_b, angle_a), (angle_b, angle_a)]
    # z_max = 10

    # for ax, specific_variance, view_angle in zip(axes, variances_to_plot, view_angles):
    #     for comp, color in zip(components, colors):
    #         subset = df[(np.isclose(df['Variance'], specific_variance, atol=1e-6)) & (df['Components'] == comp)]

    #         # Pivot the data for 3D plotting
    #         pivot_table = subset.pivot(index='Width', columns='Depth', values='Mean_Rel')

    #         # Create meshgrid for plotting
    #         X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    #         Z = pivot_table.values

    #         # Plot the 3D surface
    #         ax.plot_surface(X, Y, Z, color=color, edgecolor='k', alpha=0.5, label=f'Components={comp}')
    #         ax.set_zlim(0, 1)

    #         print(f"Components: {comp}")

    #         # print("Errors")
    #         # print(f"True Mean: {subset['True_mean'].values[:]}")
    #         # print(f"True Variance: {subset['True_var'].values[:]}")
    #         # print(f"GM Man: {subset['Mean_Rel'].values[:]*subset['True_mean'].values[:]}")
    #         # print(f"GM Var: {subset['Var_Rel'].values[:]*subset['True_var'].values[:]}")
    #         # print(f"Mean Rel: {subset['Mean_Rel'].values[:]}")
    #         # print(f"Var Rel: {subset['Var_Rel'].values[:]}")


    #     ax.set_xlabel('Depth')
    #     ax.set_ylabel('Width')
    #     ax.set_zlabel('Rel. Mean Error')
    #     ax.set_title(f'Mean Error Surface Plot\n(Variance={specific_variance})')
    #     ax.legend([f'Components={comp}' for comp in components], loc='upper left', title='Components')

    #     # Set the initial view angle
    #     ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # plt.tight_layout()
    # plt.savefig("workbench/mean_architecture_study_0.png", dpi=300)

    # #****** Do the Var surface Plots
    # fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw={'projection': '3d'})
    # axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # # Variances to plot
    # variances_to_plot = [0.1, 0.5, 1.0, 1.5]
    # components = [1, 3, 5, 9, 15]
    # colors = ['red', 'blue', 'green', 'yellow','purple']  # Constant colors for each component

    # # Define initial view angles for each subplot
    # angle_a = 220
    # angle_b = 5
    # view_angles = [(angle_b, angle_a), (angle_b, angle_a), (angle_b, angle_a), (angle_b, angle_a)]
    # z_max = 10

    # for ax, specific_variance, view_angle in zip(axes, variances_to_plot, view_angles):
    #     for comp, color in zip(components, colors):
    #         subset = df[(np.isclose(df['Variance'], specific_variance, atol=1e-6)) & (df['Components'] == comp)]

    #         # Pivot the data for 3D plotting
    #         pivot_table = subset.pivot(index='Width', columns='Depth', values='Var_Rel')

    #         # Create meshgrid for plotting
    #         X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    #         Z = pivot_table.values

    #         # Plot the 3D surface
    #         ax.plot_surface(X, Y, Z, color=color, edgecolor='k', alpha=0.5, label=f'Components={comp}')
    #         ax.set_zlim(0, 1)

    #         print("Errors")
    #         print(f"True Mean: {subset['True_mean'].values[:]}")
    #         print(f"True Variance: {subset['True_var'].values[:]}")
    #         print(f"GM Man: {subset['Mean_Rel'].values[:]*subset['True_mean'].values[:]}")
    #         print(f"GM Var: {subset['Var_Rel'].values[:]*subset['True_var'].values[:]}")
    #         print(f"Mean Rel: {subset['Mean_Rel'].values[:]}")
    #         print(f"Var Rel: {subset['Var_Rel'].values[:]}")


    #     ax.set_xlabel('Depth')
    #     ax.set_ylabel('Width')
    #     ax.set_zlabel('Rel. Var Error')
    #     ax.set_title(f'Var Error Surface Plot\n(Variance={specific_variance})')
    #     ax.legend([f'Components={comp}' for comp in components], loc='upper left', title='Components')

    #     # Set the initial view angle
    #     ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # plt.tight_layout()
    # plt.savefig("workbench/var_architecture_study_0.png", dpi=300)

    # plt.show()
    
    # # Filter data for specific depths and widths
    # depths_to_plot = [1, 3, 5 , 10, 20, 30]
    # widths_to_plot = [1, 3, 5 , 10, 20, 30]

    # #******Create Plot for widths******
    # specified_variance = 1.0  # Specify the variance you want to plot

    # for specified_depth in depths_to_plot:
    #     plt.figure(figsize=(10, 8))

    #     for width in widths_to_plot:
    #         subset = df[(np.isclose(df['Depth'], specified_depth, atol=1e-6)) & 
    #                 (np.isclose(df['Width'], width, atol=1e-6)) & 
    #                 (np.isclose(df['Variance'], specified_variance, atol=1e-6))]
    #         subset = subset.sort_values(by='Components', ascending=True)
    #         plt.plot(subset['Components'], subset['Mean_Rel'], label=f'Depth={specified_depth}, Width={width}, Variance={specified_variance}', marker='o')

    #     plt.xlabel('Components')
    #     plt.ylabel('Relative Mean Error')
    #     plt.ylim(0,1)
    #     plt.title('Mean Error vs Components for Depth and Width (1 to 5)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(f"workbench/mean_architecture_widths_d{specified_depth}_v{specified_variance}.png", dpi=300)

    #     # Filter data for specific depths and widths
    # depths_to_plot = [1, 3, 5 , 10, 20, 30]
    # widths_to_plot = [1, 3, 5 , 10, 20, 30]

    # #Create plots for depths
    # specified_variance = 1.0  # Specify the variance you want to plot

    # for specified_width in widths_to_plot:
    #     plt.figure(figsize=(10, 8))
    #     for depth in depths_to_plot:
    #         subset = df[(np.isclose(df['Width'], specified_width, atol=1e-6)) & 
    #                 (np.isclose(df['Depth'], depth, atol=1e-6)) & 
    #                 (np.isclose(df['Variance'], specified_variance, atol=1e-6))]
    #         subset = subset.sort_values(by='Components', ascending=True)
    #         plt.plot(subset['Components'], subset['Mean_Rel'], label=f'Width={specified_width}, Depth={depth}, Variance={specified_variance}', marker='o')

    #     plt.xlabel('Components')
    #     plt.ylabel('Relative Mean Error')
    #     plt.ylim(0,1)
    #     plt.title('Mean Error vs Components for Width and Depth (1 to 5)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(f"workbench/mean_architecture_depths_w{specified_width}_v{specified_variance}.png", dpi=300)

    # print(f"Shape of training points: {len(training_pts_in)}")


    subset = df[(np.isclose(df['Variance'], 1.0, atol=1e-6))]
    # Compute Width/Depth Parameter and add it to every line
    subset.loc[:, 'WD_Ratio'] = subset['Width'] / subset['Depth']  # Safely add the new column

    # Ensure all Mean_Rel values are positive
    subset['Mean_Rel'] = subset['Mean_Rel'].abs()

    subset = subset.sort_values(by='WD_Ratio', ascending=True)
    subset = subset.reset_index(drop=True)  # Reset the index after sorting

    plt.figure(figsize=(10, 6))
    gm_1= subset[subset['Components'] == 1]
    plt.scatter(gm_1['WD_Ratio'], gm_1['Mean_Rel'], label='1 component', color='red', marker='o')
    gm_2 = subset[subset['Components'] == 2]
    plt.scatter(gm_2['WD_Ratio'], gm_2['Mean_Rel'], label='2 components', color='blue', marker='o')
    gm_5 = subset[subset['Components'] == 5]
    plt.scatter(gm_5['WD_Ratio'], gm_5['Mean_Rel'], label='5 components', color='green', marker='o')
    gm_10 = subset[subset['Components'] == 10]
    plt.scatter(gm_10['WD_Ratio'], gm_10['Mean_Rel'], label='10 components', color='yellow', marker='o')
    gm_15 = subset[subset['Components'] == 15]
    plt.scatter(gm_15['WD_Ratio'], gm_15['Mean_Rel'], label='15 components', color='purple', marker='o')
    plt.axhline(y=0.1, color='black', linestyle='--', linewidth=1, label='10%')
    plt.xlabel('Width/Depth Ratio')
    plt.ylabel('Relative Mean Error')
    plt.title("Mean Error vs Width/Depth Ratio")
    plt.ylim(0,2)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.savefig("workbench/ratio_vs_mean_error.png", dpi=300)


if single_run:
    # Set Parametrrs
    num_samples = 150000
    var = 1.0
    width = 2
    depth = 2
    components = 4

    layers = [1] + [width]*depth + [1] # Add the input and output layer
    weights = generate_weights(layers,num_samples,var) # Generate weights for the network
    #Generate input (standard Normal)
    z_0 = torch.randn(num_samples,1,1,device=device) # Input to the network (samples x in_neurons x batch)

    _,_,z_hist,z_gm_hist = network(z_0,layers,[components]*(len(layers)-1),weights,single_run=True) # Compute the forward pass

    #******Viualize******
    # Initialize a plot with depth plots stacked on top of each other
    fig, axes = plt.subplots(len(z_hist), 1, figsize=(10, 5 * len(z_hist)))

    # Ensure axes is iterable even if there's only one subplot
    if len(z_hist) == 1:
        axes = [axes]

    # Plot KDE for each depth
    for i, (z, z_gm, ax) in enumerate(zip(z_hist, z_gm_hist, axes)):
        # Choose which parameter to plot
        z = z[:,0,0]
        z_flat = z.numpy().flatten()  # Flatten the tensor for KDE
        sns.kdeplot(z_flat, ax=ax, fill=True, color="blue", alpha=0.5, label=f"z_hist, mean: {np.mean(z_flat):.4f}", bw_adjust=2)
        #ax.scatter(z_flat, np.zeros_like(z_flat), color="blue", alpha=0.5, s=50, label="z_hist points")

        z_gm = z_gm[:,0,0]
        z_gm_flat = z_gm.numpy().flatten()  # Flatten the tensor for KDE
        sns.kdeplot(z_gm_flat, ax=ax, fill=True, color="orange", alpha=0.5, label=f"z_gm_hist, mean: {np.mean(z_gm_flat):.4f}", bw_adjust=2)
        #ax.scatter(z_gm_flat, np.zeros_like(z_gm_flat), color="orange", alpha=0.5, s=50, label="z_gm_hist points")

        ax.set_title(f"KDE of z_hist[{i}] and z_gm_hist[{i}] (Layer {i+1})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        # if not i == 0:
        #     ax.set_xlim(ax.get_xlim()[0] * 0.1, ax.get_xlim()[1] * 0.1)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("workbench/kde_z_hist_with_gm.png", dpi=300)
