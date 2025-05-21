import numpy as np
import matplotlib.pyplot as plt
import GaussianMixtureNetwork as GMN
from sklearn.mixture import GaussianMixture
import seaborn as sns

#******Testbench******

# Check the moment matching function for two gaussians
# Explicitly set the parameters of the 4-component scalar Gaussian mixture model

# Define weights, means, and covariances
weights = np.array([0.25, 0.25, 0.25, 0.25])
means = np.array([[0.0], [1], [8.0], [9.0]])
covariances = np.array([[[1.0]], [[1.0]], [[1.0]], [[1.0]]])*0.1

# Create and set up the GMM
gmm = GaussianMixture(n_components=4, covariance_type='full')
gmm.weights_ = weights
gmm.means_ = means
gmm.covariances_ = covariances

# Samplepoints from the specified GMM
samples, _ = gmm.sample(10000)

# Computze empirical moments op to order 10
moments = []
for i in range(1,11):
    moments.append(np.mean(samples**i))

# Try the moment matching function
mu,c,w = GMN.match_moments_2(moments,2)

# Make another GMM and set the parameters according to the result
gmm2 = GaussianMixture(n_components=2, covariance_type='full')
gmm2.weights_ = w
gmm2.means_ = mu.reshape(-1, 1)
gmm2.covariances_ = c.reshape(-1, 1, 1)

# Samplepoints from the GMM
samples2, _ = gmm2.sample(10000)

# Plot the samples using seaborn KDE
sns.kdeplot(samples.flatten(), fill=True, color='g', alpha=0.6, label='Original GMM', bw_adjust=0.1)
sns.kdeplot(samples2.flatten(), fill=True, color='r', alpha=0.6,label='Fitted GMM')
plt.title("KDE of Samples from Explicit 4-component Gaussian Mixture")
plt.xlabel("Value")
plt.ylabel("Density")


# Check the analytic moment computation of the combined det/gauss Pre activation
means = np.array([0.2, 0.1, 0.05, 0.6, 0.9])
vars = np.array([1.0, 1.0, 1.0, 1.0, 01.0])

x_new = np.array([3.0, 4.0, 1.0, 2.0, 5.0])

x_gm = np.array([
    [[x_new[0]], [0], [1]],
    [[x_new[1]], [0], [1]],
    [[x_new[2]], [0], [1]],
    [[x_new[3]], [0], [1]],
    [[x_new[4]], [0], [1]]
])

# Sample based reference
num_samples = 100000
samples_gauss1 = np.random.normal(means[0], np.sqrt(vars[0]), num_samples)
samples_gauss2 = np.random.normal(means[1], np.sqrt(vars[1]), num_samples)
samples_gauss3 = np.random.normal(means[2], np.sqrt(vars[2]), num_samples)
samples_gauss4 = np.random.normal(means[3], np.sqrt(vars[3]), num_samples)
samples_gauss5 = np.random.normal(means[4], np.sqrt(vars[4]), num_samples)
samples_result = x_new[0]*samples_gauss1 + x_new[1]*samples_gauss2 + x_new[2]*samples_gauss3+ x_new[3]*samples_gauss4 + x_new[4]*samples_gauss5

samples_moments = []
for i in range(1,11):
    samples_moments.append(np.mean(samples_result**i))

# Evaluate target function
result = GMN.moments_pre_act_combined_general(x_gm,np.stack((means,vars),axis=1))

print()
print("COMPARISON (Det/gauss) combined:")
for i in range(10):
    print(f"Sampled: {samples_moments[i]:.4f}, Analytic: {result[i]:.4f}, Rel. Err.: {100*abs((samples_moments[i]-result[i])/samples_moments[i]):.4f} %")



# # Define parameters
# layers = [1,10,1]
# act_func = ['relu','linear']

# # Build Network
# model = GMN.GaussianMixtureNetwork(layers,act_func,2,2,0.05)

# model.print_network()

# model.forward_moments(2.0)