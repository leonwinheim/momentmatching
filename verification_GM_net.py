import numpy as np
import matplotlib.pyplot as plt
import GaussianMixtureNetwork as GMN
from sklearn.mixture import GaussianMixture

verify_pre_act = False
verify_post_act = False

# Control variables
np.random.seed(4)  # Set seed for reproducibility

if verify_pre_act:
    #******Verify the function of the analytical moment computation******
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

layers = [1,2,2,1]
act_func = ['relu','relu','linear']

print("Initializing model...")
model = GMN.GaussianMixtureNetwork(layers,act_func,2,2,0.05)
print("Initialized model.")
model.print_network()
print()
result = model.forward_samples(np.array([[2.0]]))
result = model.forward_moments(np.array([[2.0]]))

for i in range(len(model.pre_activation_moments_analytic)):
    rel_pre = np.round(100*(model.pre_activation_moments_analytic[i]-model.pre_activation_moments_samples[i])/model.pre_activation_moments_samples[i],2)

    rel_post = np.round(100*(model.post_activation_moments_analytic[i]-model.post_activation_moments_samples[i])/model.post_activation_moments_samples[i],2)

    print(f"******Layer {i}******")
    print(f"Max. Pre activation rel. error: {np.max(abs(rel_pre))} %; Max. First Moment rel. error: {np.max(abs(rel_pre[:,0]))} %")
    print(f"Max. Post activation rel. error: {np.max(abs(rel_post))} %; Max. First Moment rel. error: {np.max(abs(rel_post[:,0]))} %")
    print()
    print()
    print("Pre:")
    print(rel_pre)
    print("Post:")
    print(rel_post)
