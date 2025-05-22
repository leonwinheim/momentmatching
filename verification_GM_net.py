###########################################################################
# This script verifies the Gaussian Mixture Network (GMN) implementation
# Author: Leon Winheim
# Date: 22.05.2025
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import GaussianMixtureNetwork as GMN
from sklearn.mixture import GaussianMixture
import seaborn as sns
import time

# Flags to chose what to do
verify_pre_act = False
verify_post_act = False
verify_moment_matching = False
verify_network = True

# Control variables
np.random.seed(4)  # Set seed for reproducibility

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
    num_samples = 500000
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
    layers = [1,5,2,5,1]
    act_func = ['relu','relu','relu','linear']
    gm_comp_pre = 2
    gm_comp_post =2

    print("Initializing model...")
    model = GMN.GaussianMixtureNetwork(layers,act_func,gm_comp_pre,gm_comp_post,0.05)
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
        figsize=(8*max(layers), 3*(len(layers)-1))  # Broadened width from 4* to 8*
    )
    for layer in range(len(layers)-1):
        for neuron in range(layers[layer+1]):
            print(f"***Layer {layer}, Neuron {neuron}***")
            # Pre Activation
            # Make the sample kde
            sns.kdeplot(
                model.pre_activation_samples[layer][:, neuron], 
                ax=axes[neuron, layer*2], 
                fill=True, 
                label='Sampled',
                color='blue',
                bw_adjust=kde_bw_adjust
            )
            # Add a vertical line at the sample mean
            sample_mean = np.mean(model.pre_activation_samples[layer][:, neuron])
            # sample_mean_moment = model.pre_activation_moments_samples[layer][neuron,0]
            # print(f"Sample vs Moment Pre-Act Sample: {sample_mean:.2f} vs {sample_mean_moment:.2f}")

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

            sns.kdeplot(
                gm_samples, 
                ax=axes[neuron, layer*2], 
                fill=True, 
                label='GM',
                color='orange',
                bw_adjust=kde_bw_adjust
            )
            # Add a vertical line at the GM mean
            gm_mean = np.mean(gm_samples)
            # gm_mean_moment = model.pre_activation_moments_analytic[layer][neuron,0]
            # gm_mean_moment_new = GMN.e1_gm(model.weights_gm_pre[layer][neuron, :], model.means_gm_pre[layer][neuron, :], model.variances_gm_pre[layer][neuron, :])

            # print(f"Sample vs Moment Pre Act GM : Samp{gm_mean:.2f} vs Soll{gm_mean_moment:.2f} vs Ist{gm_mean_moment_new:.2f}")

            axes[neuron, layer*2].axvline(
                gm_mean, 
                color='orange', 
                linestyle='--', 
                label=f'GM Mean: {gm_mean:.2f}'
            )

            axes[neuron, layer*2].set_title(f'L {layer} N {neuron} Pre-Act')
            # Post Activation
            # Make the sample kde
            sns.kdeplot(
                model.post_activation_samples[layer][:, neuron], 
                ax=axes[neuron, layer*2+1], 
                fill=True, 
                label='Sampled',
                color='blue',
                bw_adjust=kde_bw_adjust
            )
            # Add a vertical line at the sample mean
            sample_mean = np.mean(model.post_activation_samples[layer][:, neuron])
            # sample_mean_moment = model.post_activation_moments_samples[layer][neuron,0]
            # print(f"Sample vs Moment Post-Act Samples: {sample_mean:.2f} vs {sample_mean_moment:.2f}")

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

            sns.kdeplot(
                gm_samples, 
                ax=axes[neuron, layer*2+1], 
                fill=True, 
                label='GM',
                color='orange',
                bw_adjust=kde_bw_adjust
            )
            # Add a vertical line at the GM mean
            gm_mean = np.mean(gm_samples)
            # gm_mean_moment = model.post_activation_moments_analytic[layer][neuron,0]	
            # gm_mean_moment_new = GMN.e1_gm(model.weights_gm_post[layer][neuron, :], model.means_gm_post[layer][neuron, :], model.variances_gm_post[layer][neuron, :])

            # print(f"Sample vs Moment Post Act GM :Samp{gm_mean:.2f} vs Soll{gm_mean_moment:.2f} vs Ist{gm_mean_moment_new:.2f}")

            axes[neuron, layer*2+1].axvline(
                gm_mean, 
                color='orange', 
                linestyle='--', 
                label=f'GM Mean: {gm_mean:.2f}'
            )

            axes[neuron, layer*2+1].set_title(f'L {layer} N{neuron} Post-Act')

            #Activate legends
            axes[neuron, layer*2].legend(loc='upper right')
            axes[neuron, layer*2+1].legend(loc='upper right')


    plt.tight_layout()
    plt.savefig('figures/GMN_verification.png', dpi=300)
