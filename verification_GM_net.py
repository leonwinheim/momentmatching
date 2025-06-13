###########################################################################
# This script verifies the Gaussian Mixture Network (GMN) implementation
# Author: Leon Winheim
# Date: 22.05.2025
###########################################################################
print("Startet")
import numpy as np
print("Numpy")
import matplotlib.pyplot as plt
print("Matplotlib")
import GaussianMixtureNetwork as GMN
print("GaussianMixtureNetwork")
from sklearn.mixture import GaussianMixture
import seaborn as sns
import time
import os
from scipy.special import erf
import pickle
import pandas as pd
import math

print("Rennt..")

os.environ["LOKY_MAX_CPU_COUNT"] = "10"

# Flags to chose what to do
verify_pre_act = False
verify_post_act = False
verify_moment_matching = False
verify_network = True
verify_moment_spike = False
verify_gauss_moment = False
verify_network_peak = False
verify_moment_matching_peak = False
test_tuples = False
test_factorial = False

# Control variables
np.random.seed(4)  # Set seed for reproducibility

# Own Functions
def sample_from_gmm(weights, means, variances, n_samples):
    # Step 1: Choose components according to their weights
    component_choices = np.random.choice(len(weights), size=n_samples, p=weights)
    
    # Step 2: Sample from the corresponding Gaussians
    samples = np.random.normal(loc=np.array(means)[component_choices],
                               scale=np.array(np.sqrt(variances))[component_choices])
    return samples

def sample_from_dirac_gauss_mixture(dirac_weights, dirac_locs, gauss_weights, gauss_means, gauss_sigmas, n_samples):
    # Combine all component weights
    weights = np.concatenate([dirac_weights, gauss_weights])
    
    # Ensure weights sum to 1 (optional, but good practice)
    weights = weights / np.sum(weights)
    
    n_diracs = len(dirac_weights)
    n_gaussians = len(gauss_weights)
    
    # Choose which component to sample from
    component_choices = np.random.choice(n_diracs + n_gaussians, size=n_samples, p=weights)
    
    # Prepare the samples array
    samples = np.empty(n_samples)
    
    # Sample from Dirac components
    for i, loc in enumerate(dirac_locs):
        indices = (component_choices == i)
        samples[indices] = loc
    
    # Sample from Gaussian components
    for j, (mu, sigma) in enumerate(zip(gauss_means, gauss_sigmas)):
        indices = (component_choices == n_diracs + j)
        samples[indices] = np.random.normal(loc=mu, scale=sigma, size=np.sum(indices))
    
    return samples

if verify_pre_act:
    #******Verify the function of the analytical moment computation******
    print("Verifying the function of the analytical moment computation...")

    num_input = 10

    # Define Deterministic inputs
    x_input = np.zeros((num_input,3,1))     #Input
    w = np.zeros((num_input,2))       #Weights
    for i in range(num_input):
        x_input[i,:,:] = np.array([[np.random.rand()],[0.0],[1.0]])
        w[i,:] = np.array([np.random.rand(),1.0])

    # Draw samples from input and weights
    num_samples = 1000000
    w_samples = np.zeros((num_samples,num_input))
    for i in range(num_input):
        w_samples[:,i] = np.random.normal(w[i,0],w[i,1],num_samples)

    # Compute the transformed samples
    a = np.zeros(num_samples)
    for i in range(num_input):
        a += w_samples[:,i]*x_input[i,0,0]

    # Compute empirical moments of the samples up to order 10
    empirical_moments = []
    for i in range(1,11):
        empirical_moments.append(np.mean(a**i))

    # Compute the analytic moments
    analytic_moments = GMN.moments_pre_act_combined_general(x_input,w)

    # Compute the relative error
    rel_error = np.round(100*(analytic_moments-empirical_moments)/empirical_moments,2)
    print("******Verification of the analytic moment computation******")
    for i in range(10):
        print(f"Sampled: {empirical_moments[i]:.4f}, Analytic: {analytic_moments[i]:.4f}, Rel. Err.: {rel_error[i]:.4f} %")

    # Define True GM input parameters
    x_gm_input = np.zeros((num_input,3,2))     #Input
    for i in range(num_input):
        x_gm_input[i,:,:] = np.array([[np.random.rand(),np.random.rand()],[1.0,1.5],[0.3,0.7]])

    # Sample from multiple GMs
    x_gm_samples = np.zeros((num_samples,num_input))
    for i in range(num_input):
        gm = GaussianMixture(n_components=2)
        # Prepare means and covariances in the expected shapes
        gm.means_ = x_gm_input[i,0,:].reshape(-1,1)
        gm.covariances_ = x_gm_input[i,1:,:].reshape(-1,1,1)
        gm.weights_ = x_gm_input[i,2,:]
        x_gm_samples[:,i] = gm.sample(num_samples)[0].reshape(-1)

    # Compute the transformed samples
    a_gm = np.zeros(num_samples)
    for i in range(num_input):
        a_gm += w_samples[:,i]*x_gm_samples[:,i]

    # Compute empirical moments of the samples up to order 10
    empirical_moments_gm = []
    for i in range(1,11):
        empirical_moments_gm.append(np.mean(a_gm**i))

    # Compute the analytic moments
    analytic_moments_gm = GMN.moments_pre_act_combined_general(x_gm_input,w)

    # Compute the relative error
    rel_error_gm = np.round(100*(analytic_moments_gm-empirical_moments_gm)/empirical_moments_gm,2)

    print()
    print("******Verification of the analytic moment computation for GM inputs******")
    for i in range(10):
        print(f"Sampled: {empirical_moments_gm[i]:.4f}, Analytic: {analytic_moments_gm[i]:.4f}, Rel. Err.: {rel_error_gm[i]:.4f} %")

if verify_post_act:
    #******Verify the post activation moment computation******
    print()
    print("Verifying the post activation moment computation...")

    # Generate samples from an arbitrary input GM Distribution with two components
    num_samples = 100000

    means = np.array([0.2, 2.1])
    vars = np.array([1.0, 1.0])
    weights = np.array([0.5, 0.5])
    gm = GaussianMixture(n_components=2, covariance_type='full')
    gm.weights_ = weights
    gm.means_ = means.reshape(-1, 1)
    gm.covariances_ = vars.reshape(-1, 1, 1)
    input_samples, _ = gm.sample(num_samples)

    # Define leaky reLu
    alpha = 0.1
    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)

    # Compute the transformed samples
    z_samples = leaky_relu(input_samples)

    # Compute empirical moments of the samples up to order 10
    empirical_moments = []
    for i in range(1,11):
        empirical_moments.append(np.mean(z_samples**i))
    # Compute the analytic moments
    analytic_moments = GMN.moments_post_act(alpha,means,vars,weights)

    # Compute the relative error
    rel_error = np.round(100*(analytic_moments-empirical_moments)/empirical_moments,2)

    print()
    print("******Verification of the analytic moment computation for GM inputs after leaky ReLU******")
    for i in range(10):
        print(f"Sampled: {empirical_moments[i]:.4f}, Analytic: {analytic_moments[i]:.4f}, Rel. Err.: {rel_error[i]:.4f} %")

if verify_moment_matching:
    #******Verify the moment matching function******
    print()
    print("Verifying the moment matching function...")
    # Generate a 3 Component Gaussian Mixture with arbitrary means, vars and weights
    n_components = 3
    means = np.array([-4, 1, 2])  # scalar means
    variances = np.array([0.1, 0.5, 0.1])  # scalar variances
    weights = np.array([1 / n_components] * n_components)  # equal weights
    n_samples = 100000

    # Create the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components)
    gmm.means_ = means.reshape(-1, 1)
    gmm.covariances_ = variances.reshape(-1, 1, 1)
    gmm.weights_ = weights

    # Sample from the GMM
    X = gmm.sample(n_samples)[0]

    # Compute the first ten empirical moments
    def empirical_moments(X, n_moments):
        moments = []
        for i in range(n_moments):
            moment = np.mean(X ** (i + 1))
            moments.append(moment)
        return np.array(moments)

    n_moments = 10
    empirical_moments_X = empirical_moments(X, n_moments)

    # Try fo fit the moment matching func
    components_fit = 2
    mu_fit, c_fit, w_fit = GMN.match_moments(empirical_moments_X, components_fit)

    moments = [GMN.e1_gm(w_fit, mu_fit, c_fit),
                GMN.e2_gm(w_fit, mu_fit, c_fit),
                GMN.e3_gm(w_fit, mu_fit, c_fit),
                GMN.e4_gm(w_fit, mu_fit, c_fit),
                GMN.e5_gm(w_fit, mu_fit, c_fit),
                GMN.e6_gm(w_fit, mu_fit, c_fit),
                GMN.e7_gm(w_fit, mu_fit, c_fit),
                GMN.e8_gm(w_fit, mu_fit, c_fit),
                GMN.e9_gm(w_fit, mu_fit, c_fit),
                GMN.e10_gm(w_fit, mu_fit, c_fit)]


    print("Fitted means:", mu_fit)
    print("Fitted covariances:", c_fit)
    print("Fitted weights:", w_fit)
    print("***")
    print("Empirical vs Fitted moments")
    for i in range(n_moments):
        print(f"Moment {i+1}: Empirical: {empirical_moments_X[i]:.4f}, Fitted: {moments[i]:.4f}, relative error: {100*abs(empirical_moments_X[i] - moments[i]) / abs(empirical_moments_X[i]):.2f}%")

    #Sample from a GM with the fitted parameters
    gmm_fit = GaussianMixture(n_components=components_fit)
    gmm_fit.means_ = mu_fit.reshape(-1, 1)
    gmm_fit.covariances_ = c_fit.reshape(-1, 1, 1)
    gmm_fit.weights_ = w_fit
    X_fit = gmm_fit.sample(n_samples)[0]

    ## Make a kdensity plot of the samples
    sns.kdeplot(X.flatten(),fill=True, bw_adjust=1.0, color='blue', alpha=0.5)
    sns.kdeplot(X_fit.flatten(),fill=True, bw_adjust=1.0, color='red', alpha=0.5)
    plt.title("KDE of samples from GMM")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.savefig("kde.png")

if verify_network:
    layers = [1,5,5,1]
    act_func = ['relu','relu','linear']
    gm_comp_pre = 2
    gm_comp_post = 2
    moments_pre = 5
    moments_post = 5

    print("Initializing model...")
    model = GMN.GaussianMixtureNetwork(layers,act_func,gm_comp_pre,gm_comp_post,moments_pre,moments_post,0.00)
    print("Initialized model.")
    model.print_network()
    print()

    x = np.array([[2.0]])

    #Call comparison function
    model.compare_sample_moments_forward(x)
    #model.compare_sample_moments_forward_special(x)

    #******Visualization******
    num_samples = 10000

    # Set kernel width multiplier for all kdeplots
    kde_bw_adjust = 0.5  # You can adjust this value as needed
    print()
    print("Make figure...")
    print()
    fig, axes = plt.subplots(
        ncols=(len(layers)-1)*2, 
        nrows=max(layers), 
        figsize=(8*max(layers), 6*(len(layers)-1))  # Broadened width from 4* to 8*
    )
    for layer in range(len(layers)-1):
        for neuron in range(layers[layer+1]):
            print(f"***Layer {layer}, Neuron {neuron}***")
            # Pre Activation
            axes[neuron, layer*2].hist(
                model.pre_activation_samples[layer][:, neuron],
                bins=50,
                density=True,
                alpha=0.6,
                color='blue',
                label='Sampled'
            )
            # Add a vertical line at the sample mean
            sample_mean = np.mean(model.pre_activation_samples[layer][:, neuron])

            axes[neuron, layer*2].axvline(
                sample_mean, 
                color='blue', 
                linestyle='--', 
                label=f'Mean: {sample_mean:.2f}'
            )
            # Generate and plot GM samples from the moment matching
            gm = GaussianMixture(n_components=gm_comp_pre)
            gm.means_ = model.means_gm_pre[layer][neuron, :].reshape(-1, 1)
            gm.covariances_ = model.variances_gm_pre[layer][neuron, :].reshape(-1, 1, 1)
            gm.weights_ = model.weights_gm_pre[layer][neuron, :]

            gm_samples = gm.sample(num_samples)[0].flatten()

            axes[neuron, layer*2].hist(
                gm_samples,
                bins=50,
                density=True,
                alpha=0.6,
                color='orange',
                label='GM'
            )
            # Add a vertical line at the GM mean
            gm_mean = np.mean(gm_samples)

            axes[neuron, layer*2].axvline(
                gm_mean, 
                color='orange', 
                linestyle='--', 
                label=f'GM Mean: {gm_mean:.2f}'
            )

            # Generate and plot EM-GM samples from the moment matching
            gm = GaussianMixture(n_components=gm_comp_pre)
            gm.means_ = model.means_gm_pre_em[layer][neuron, :].reshape(-1, 1)
            gm.covariances_ = model.variances_gm_pre_em[layer][neuron, :].reshape(-1, 1, 1)
            gm.weights_ = model.weights_gm_pre_em[layer][neuron, :]

            gm_samples_em = gm.sample(num_samples)[0].flatten()

            axes[neuron, layer*2].hist(
                gm_samples_em,
                bins=50,
                density=True,
                alpha=0.6,
                color='green',
                label='EM-GM'
            )
            # Add a vertical line at the GM mean
            gm_mean_em = np.mean(gm_samples_em)

            axes[neuron, layer*2].axvline(
                gm_mean_em, 
                color='green', 
                linestyle='--', 
                label=f'EM-GM Mean: {gm_mean_em:.2f}'
            )

            axes[neuron, layer*2].set_title(f'L {layer} N {neuron} Pre-Act')

            # Post Activation
            axes[neuron, layer*2+1].hist(
                model.post_activation_samples[layer][:, neuron],
                bins=50,
                density=True,
                alpha=0.6,
                color='blue',
                label='Sampled'
            )
            # Add a vertical line at the sample mean
            sample_mean = np.mean(model.post_activation_samples[layer][:, neuron])

            axes[neuron, layer*2+1].axvline(
                sample_mean, 
                color='blue', 
                linestyle='--', 
                label=f'Mean: {sample_mean:.2f}'
            )

            # Generate and plot GM samples from the moment matching
            gm = GaussianMixture(n_components=gm_comp_post)
            gm.means_ = model.means_gm_post[layer][neuron, :].reshape(-1, 1)
            gm.covariances_ = model.variances_gm_post[layer][neuron, :].reshape(-1, 1, 1)
            gm.weights_ = model.weights_gm_post[layer][neuron, :]

            gm_samples = gm.sample(num_samples)[0].flatten()

            axes[neuron, layer*2+1].hist(
                gm_samples,
                bins=50,
                density=True,
                alpha=0.6,
                color='orange',
                label='GM'
            )
            # Add a vertical line at the GM mean
            gm_mean = np.mean(gm_samples)

            axes[neuron, layer*2+1].axvline(
                gm_mean, 
                color='orange', 
                linestyle='--', 
                label=f'GM Mean: {gm_mean:.2f}'
            )

            # Generate and plot EM-GM samples from the moment matching
            gm = GaussianMixture(n_components=gm_comp_post)
            gm.means_ = model.means_gm_post_em[layer][neuron, :].reshape(-1, 1)
            gm.covariances_ = model.variances_gm_post_em[layer][neuron, :].reshape(-1, 1, 1)
            gm.weights_ = model.weights_gm_post_em[layer][neuron, :]

            gm_samples_em = gm.sample(num_samples)[0].flatten()

            axes[neuron, layer*2+1].hist(
                gm_samples_em,
                bins=50,
                density=True,
                alpha=0.6,
                color='green',
                label='EM-GM'
            )
            # Add a vertical line at the GM mean
            gm_mean_em = np.mean(gm_samples_em)

            axes[neuron, layer*2+1].axvline(
                gm_mean_em, 
                color='green', 
                linestyle='--', 
                label=f'EM-GM Mean: {gm_mean_em:.2f}'
            )

            axes[neuron, layer*2+1].set_title(f'L {layer} N{neuron} Post-Act')

            #Activate legends
            axes[neuron, layer*2].legend(loc='upper right')
            axes[neuron, layer*2+1].legend(loc='upper right')


    plt.tight_layout()
    plt.savefig('figures/GMN_verification.png', dpi=300)

if verify_moment_spike:
    # Generate Samples from a Gaussian 
    num_components = 3
    num_samples = 1000000
    mu = 0.0
    sigma = 1.0
    samples = np.random.normal(mu, sigma, num_samples)

    # Propagate the samples through a leaky ReLU
    alpha = 0.1
    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
    z_samples = leaky_relu(samples, alpha)

    # Compute empirical moments of the samples up to order 10
    empirical_moments = []
    for i in range(1, 11):
        empirical_moments.append(np.mean(z_samples ** i))

    # Compute the analytic moments
    analytic_moments = GMN.moments_post_act(alpha, np.array([[mu]]), np.array([[sigma**2]]), np.array([[1.0]]))

    # Compute the relative error
    rel_error = np.round(100 * (analytic_moments - empirical_moments) / empirical_moments, 2)
    print()
    print("******Verification of the analytic moment computation for leaky ReLU******")
    for i in range(10):
        print(f"Sampled: {empirical_moments[i]:.4f}, Analytic: {analytic_moments[i]:.4f}, Rel. Err.: {rel_error[i]:.4f} %")

    # Fit the GM to the propagated Samples
    mu_res, c_res, w_res = GMN.match_moments(analytic_moments, num_components)
    print()
    print("Resulting GM parameters")
    print("Fitted means:", mu_res)
    print("Fitted covariances:", c_res)
    print("Fitted weights:", w_res)
    moments = [GMN.e1_gm(w_res, mu_res, c_res),
               GMN.e2_gm(w_res, mu_res, c_res),
               GMN.e3_gm(w_res, mu_res, c_res),
               GMN.e4_gm(w_res, mu_res, c_res),
               GMN.e5_gm(w_res, mu_res, c_res),
               GMN.e6_gm(w_res, mu_res, c_res),
               GMN.e7_gm(w_res, mu_res, c_res),
               GMN.e8_gm(w_res, mu_res, c_res),
               GMN.e9_gm(w_res, mu_res, c_res),
               GMN.e10_gm(w_res, mu_res, c_res)]

    # Compute the relative error
    rel_error_gm = np.round(100 * (moments - analytic_moments) / analytic_moments, 2)
    print()
    print("******Comparison of analytic vs. GM-fitted moments for leaky ReLU******")
    for i in range(10):
        print(f"Analytic: {analytic_moments[i]:.4f}, GM: {moments[i]:.4f}, Rel. Err.: {rel_error_gm[i]:.4f} %")

    # Sample from a GM with the fitted parameters
    gmm_fit = GaussianMixture(n_components=num_components)
    gmm_fit.means_ = mu_res.reshape(-1, 1)
    gmm_fit.covariances_ = c_res.reshape(-1, 1, 1)
    gmm_fit.weights_ = w_res
    X_fit = gmm_fit.sample(num_samples)[0]

    # Make a kdeplot of the samples
    plt.figure(figsize=(7, 5))
    plt.title("Result of Moment matching")
    #sns.kdeplot(X_fit.flatten(), fill=True, bw_adjust=0.5, color='blue', alpha=0.5, label='GM Samples')
    #sns.kdeplot(z_samples.flatten(), fill=True, bw_adjust=0.5, color='green', alpha=0.5, label='Leaky ReLU Samples')
    plt.hist(X_fit.flatten(), bins=100, density=True, color='blue', alpha=0.5, label='GM Samples')
    plt.hist(z_samples.flatten(), bins=100, density=True, color='green', alpha=0.5, label='Leaky ReLU Samples')
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Freq")
    plt.tight_layout()
    plt.savefig('figures/moment_matching_3comp.png', dpi=300)

if verify_network_peak:
    layers = [1,5,5,1]
    act_func = ['relu','relu','linear']
    gm_comp_pre = 2
    gm_comp_post = 2
    moments_pre = 10
    moments_post = 10
    peak = False

    print("Initializing model...")
    model = GMN.GaussianMixtureNetwork(layers,act_func,gm_comp_pre,gm_comp_post,moments_pre,moments_post,0.00,peak=peak)
    print("Initialized model.")
    model.print_network()
    print()

    x = np.array([[2.0]])

    #Call comparison function
    model.compare_sample_moments_forward(x)

    #******Visualization******
    num_samples = 10000

    # Set kernel width multiplier for all kdeplots
    kde_bw_adjust = 0.5  # You can adjust this value as needed
    print()
    print("Make figure...")
    print()
    fig, axes = plt.subplots(
        ncols=(len(layers)-1)*2, 
        nrows=max(layers), 
        figsize=(8*max(layers), 6*(len(layers)-1))  # Broadened width from 4* to 8*
    )
    for layer in range(len(layers)-1):
        for neuron in range(layers[layer+1]):
            print(f"***Layer {layer}, Neuron {neuron}***")
            # Pre Activation
            axes[neuron, layer*2].hist(
                model.pre_activation_samples[layer][:, neuron],
                bins=50,
                density=True,
                alpha=0.6,
                color='blue',
                label='Sampled'
            )
            # Add a vertical line at the sample mean
            sample_mean = np.mean(model.pre_activation_samples[layer][:, neuron])

            axes[neuron, layer*2].axvline(
                sample_mean, 
                color='blue', 
                linestyle='--', 
                label=f'Mean: {sample_mean:.2f}'
            )
            # Generate and plot GM samples from the moment matching

            # Sample from two independent Gaussians and append the samples to a list
            means = model.means_gm_pre[layer][neuron, :]
            variances = model.variances_gm_pre[layer][neuron, :]
            weights = model.weights_gm_pre[layer][neuron, :]
            gm_samples = sample_from_gmm(weights, means, variances, num_samples)

            axes[neuron, layer*2].hist(
                gm_samples,
                bins=50,
                density=True,
                alpha=0.6,
                color='orange',
                label='GM'
            )
            # Add a vertical line at the GM mean
            gm_mean = np.mean(gm_samples)

            axes[neuron, layer*2].axvline(
                gm_mean, 
                color='orange', 
                linestyle='--', 
                label=f'GM Mean: {gm_mean:.2f}'
            )

            # Generate and plot EM-GM samples from the moment matching
            gm = GaussianMixture(n_components=gm_comp_pre)
            gm.means_ = model.means_gm_pre_em[layer][neuron, :].reshape(-1, 1)
            gm.covariances_ = model.variances_gm_pre_em[layer][neuron, :].reshape(-1, 1, 1)
            gm.weights_ = model.weights_gm_pre_em[layer][neuron, :]

            gm_samples_em = gm.sample(num_samples)[0].flatten()

            axes[neuron, layer*2].hist(
                gm_samples_em,
                bins=50,
                density=True,
                alpha=0.6,
                color='green',
                label='EM-GM'
            )
            # Add a vertical line at the GM mean
            gm_mean_em = np.mean(gm_samples_em)

            axes[neuron, layer*2].axvline(
                gm_mean_em, 
                color='green', 
                linestyle='--', 
                label=f'EM-GM Mean: {gm_mean_em:.2f}'
            )

            axes[neuron, layer*2].set_title(f'L {layer} N {neuron} Pre-Act')

            # Post Activation
            axes[neuron, layer*2+1].hist(
                model.post_activation_samples[layer][:, neuron],
                bins=50,
                density=True,
                alpha=0.6,
                color='blue',
                label='Sampled'
            )
            # Add a vertical line at the sample mean
            sample_mean = np.mean(model.post_activation_samples[layer][:, neuron])

            axes[neuron, layer*2+1].axvline(
                sample_mean, 
                color='blue', 
                linestyle='--', 
                label=f'Mean: {sample_mean:.2f}'
            )

            # Generate and plot GM samples from the moment matching
            # Sample from two independent Gaussians and append the samples to a list
            means = model.means_gm_post[layer][neuron, :]
            variances = model.variances_gm_post[layer][neuron, :]
            weights = model.weights_gm_post[layer][neuron, :]
            if peak:
                dirac_weight = model.dirac_weight_post[layer][neuron]
                dirac_locs = 0
                gm_samples = sample_from_dirac_gauss_mixture(np.array([dirac_weight]),np.array([0]),weights,means,variances,num_samples)
            else:
                gm_samples = sample_from_gmm(weights, means, variances, num_samples)

            axes[neuron, layer*2+1].hist(
                gm_samples,
                bins=50,
                density=True,
                alpha=0.6,
                color='orange',
                label='GM'
            )
            # Add a vertical line at the GM mean
            gm_mean = np.mean(gm_samples)

            axes[neuron, layer*2+1].axvline(
                gm_mean, 
                color='orange', 
                linestyle='--', 
                label=f'GM Mean: {gm_mean:.2f}'
            )

            # Generate and plot EM-GM samples from the moment matching
            gm = GaussianMixture(n_components=gm_comp_post)
            gm.means_ = model.means_gm_post_em[layer][neuron, :].reshape(-1, 1)
            gm.covariances_ = model.variances_gm_post_em[layer][neuron, :].reshape(-1, 1, 1)
            gm.weights_ = model.weights_gm_post_em[layer][neuron, :]

            gm_samples_em = gm.sample(num_samples)[0].flatten()

            axes[neuron, layer*2+1].hist(
                gm_samples_em,
                bins=50,
                density=True,
                alpha=0.6,
                color='green',
                label='EM-GM'
            )
            # Add a vertical line at the GM mean
            gm_mean_em = np.mean(gm_samples_em)

            axes[neuron, layer*2+1].axvline(
                gm_mean_em, 
                color='green', 
                linestyle='--', 
                label=f'EM-GM Mean: {gm_mean_em:.2f}'
            )

            axes[neuron, layer*2+1].set_title(f'L {layer} N{neuron} Post-Act')

            #Activate legends
            axes[neuron, layer*2].legend(loc='upper right')
            axes[neuron, layer*2+1].legend(loc='upper right')


    plt.tight_layout()
    plt.savefig('figures/GMN_verification.png', dpi=300)

if verify_moment_matching_peak:
    #Produce some Gaussian Samples
    num_samples = 100000
    mu = 0.0
    sigma = 1.0
    samples = np.random.normal(mu, sigma, num_samples)
    # Propagate the samples through a leaky ReLU
    alpha = 0.0
    def leaky_relu_peak(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
    z_samples = leaky_relu_peak(samples, alpha)
    # Compute analytical moments of propagated density
    analytic_moments = GMN.moments_post_act(alpha, np.array([[mu]]), np.array([[sigma**2]]), np.array([[1.0]]))
    # Compute empirical moments of the samples up to order 10
    empirical_moments = []
    for i in range(1, 11):
        empirical_moments.append(np.mean(z_samples ** i))

    # Integrate the probability mass of the input gaussian up to zero
    integration_limit = 0.0
    prob_mass = 0.5 * (1 + erf((integration_limit - mu) / (sigma * np.sqrt(2))))

    # Perform Moment-matching with a peak
    num_components = 2
    mu_res, c_res, w_res = GMN.match_moments_peak(analytic_moments,prob_mass, num_components)

    mu_non, c_non, w_non = GMN.match_moments(analytic_moments, num_components)

    print("Result of GM Moment Matching with Peak")
    print("Fitted means:", mu_res)
    print("Fitted covariances:", c_res)
    print("Fitted weights:", w_res)
    print("Result of GM Moment Matching without Peak")
    print("Fitted means:", mu_non)
    print("Fitted covariances:", c_non)
    print("Fitted weights:", w_non)

    moments_peak = []
    for i in range(10):
        moments_peak.append(GMN.gm_noncentral_moment(i+1, w_res, mu_res, c_res))
    
    moments_non_peak = []
    for i in range(10):
        moments_non_peak.append(GMN.gm_noncentral_moment(i+1, w_non, mu_non, c_non))
    
    # Compute the relative error
    rel_error_peak = np.round(100 * (moments_peak - analytic_moments) / analytic_moments, 2)
    rel_error_non_peak = np.round(100 * (moments_non_peak - analytic_moments) / analytic_moments, 2)
    print()
    print("******Comparison of analytic vs. GM-fitted moments with Peak******")
    for i in range(10):
        print(f"Analytic: {analytic_moments[i]:.4f}, GM with Peak: {moments_peak[i]:.4f}, Rel. Err.: {rel_error_peak[i]:.4f} %")
    print()
    print("******Comparison of analytic vs. GM-fitted moments without Peak******")
    for i in range(10):
        print(f"Analytic: {analytic_moments[i]:.4f}, GM without Peak: {moments_non_peak[i]:.4f}, Rel. Err.: {rel_error_non_peak[i]:.4f} %")

    #Sample from a GM with the fitted parameters
    gmm_fit = GaussianMixture(n_components=num_components)
    gmm_fit.means_ = mu_res.reshape(-1, 1)
    gmm_fit.covariances_ = c_res.reshape(-1, 1, 1)
    gmm_fit.weights_ = w_res
    X_fit = gmm_fit.sample(num_samples)[0]

    #Sample from a GM with the fitted parameters without peak
    gmm_fit_non = GaussianMixture(n_components=num_components)
    gmm_fit_non.means_ = mu_non.reshape(-1, 1)
    gmm_fit_non.covariances_ = c_non.reshape(-1, 1, 1)
    gmm_fit_non.weights_ = w_non
    X_fit_non = gmm_fit_non.sample(num_samples)[0]


    #Plot the results as histograms
    plt.figure(figsize=(7, 5))
    plt.title("Result of Moment matching with Peak")
    plt.hist(X_fit.flatten(), bins=100, density=True, color='blue', alpha=0.5, label='GM Samples')
    plt.hist(X_fit_non.flatten(), bins=100, density=True, color='red', alpha=0.5, label='GM Samples without Peak')
    plt.hist(z_samples.flatten(), bins=100, density=True, color='green', alpha=0.5, label='Leaky ReLU Samples')
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Freq")
    plt.tight_layout()
    plt.savefig('figures/moment_matching_peak.png', dpi=300)

if test_tuples:
    # Test how fast tuples can be generated for maximum order of moments 5 andf ten
    n_min = 1
    n_max = 6
    m_min = 10
    m_max = 50
    m_step = 5

    result = []
    perform = True
    plot = True
    
    if perform:
        for i in range(n_min, n_max+1):
            for j in range(m_min, m_max+1,m_step):
                print(f"Testing tuples for n={i}, m={j}")
                start_time = time.time()
                tuples = np.array(tuple(GMN.generate_k_tuples_fast(i,j)))
                end_time = time.time()
                print(f"Generated {len(tuples)} tuples in {end_time - start_time:.4f} seconds")
                result.append((i, j, len(tuples), end_time - start_time))
                
                with open("workbench/tuples.pkl", "wb") as f:
                    pickle.dump(result, f)
    if plot:
        # Load results from previous runs if available
        if os.path.exists("workbench/tuples.pkl"):
            with open("workbench/tuples.pkl", "rb") as f:
                result = pickle.load(f)
        else:
            print("No previous results found")
            exit()

        # Convert result to DataFrame for easier plotting
        df = pd.DataFrame(result, columns=["n", "m", "num_tuples", "time_sec"])

        plt.figure(figsize=(8, 4))
        for n in df["n"].unique():
            subset = df[df["n"] == n]
            plt.plot(subset["m"], subset["time_sec"], marker="o", label=f"n={n} (moment order)")

        plt.xlabel("m (max components)")
        plt.ylabel("Time to generate tuples (seconds)")
        plt.title("Tuple Generation Time vs. Moment Order")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("figures/tuple_generation_time.png", dpi=300)
        plt.show()
        
if test_factorial:
    start = time.time()
    res = math.factorial(100)
    end = time.time()
    print(f"Factorial of 100: {res}, computed in {end - start:.4f} seconds")