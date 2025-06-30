import GaussianMixtureNetwork as GMN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#Samplecount
num_samples = 1000000

# Define multiple 2 component gauss mixture parametrizations with 
gm_weights = np.array([
    [0.5, 0.5],
    [0.3, 0.7],
    [0.2, 0.8],
    [0.6, 0.4],
    [0.4, 0.6]
])
gm_means = np.array([
    [3.0, 10.0],
    [0.0, 20.0],
    [5.0, 8.0],
    [2.0, 15.0],
    [7.0, 12.0]
])
gm_variances = np.array([
    [10.0, 11.0],
    [8.0, 7.0],
    [12.0, 5.0],
    [9.0, 6.0],
    [13.0, 4.0]
])

# Sample from each GMM
samples = []
for i in range(gm_weights.shape[0]):
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.weights_ = gm_weights[i]
    gmm.means_ = gm_means[i].reshape(2, 1)
    gmm.covariances_ = gm_variances[i].reshape(2, 1, 1)
    
    sample, _ = gmm.sample(num_samples)
    samples.append(sample)

# Define three different Gaussians
means = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
variances = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
# Sample from each Gaussian
gaussian_samples = []
for mean, var in zip(means, variances):
    gmm = GaussianMixture(n_components=1, covariance_type='full')
    gmm.weights_ = np.array([1.0])
    gmm.means_ = np.array([[mean]])
    gmm.covariances_ = np.array([[[var]]])
    
    sample, _ = gmm.sample(num_samples)
    gaussian_samples.append(sample)

# Plot the samples using seaborn KDE
import seaborn as sns
plt.figure(figsize=(12, 8))
for i, sample in enumerate(samples):
    sns.kdeplot(sample.flatten(), fill=True, label=f'GMM {i+1}', alpha=0.6, bw_adjust=1)
for i, sample in enumerate(gaussian_samples):
    sns.kdeplot(sample.flatten(), fill=True, label=f'Gaussian {i+1}', alpha=0.6, bw_adjust=1)
plt.title("KDE of Samples from Multiple Gaussian Mixture Models and Gaussians")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()

# Compute the Pre-Activation Samples
pre_activation_samples = samples[0]*gaussian_samples[0] + samples[1]*gaussian_samples[1] + samples[2]*gaussian_samples[2] + samples[3]*gaussian_samples[3] + samples[4]*gaussian_samples[4]

#Compute the empirical moments of the pre-activation samples
moments_pre_samples = []
for i in range(1, 6):
    moments_pre_samples.append(np.mean(pre_activation_samples**i))

# Compute the moments in analytic fashion
gm_pars = np.zeros((gm_weights.shape[0], 3, gm_weights.shape[1]))
for i in range(gm_weights.shape[0]):
    gm_pars[i, 0, :] = gm_means[i]
    gm_pars[i, 1, :] = gm_variances[i]
    gm_pars[i, 2, :] = gm_weights[i]

gauss_pars = np.zeros((means.shape[0], 2))
for i in range(means.shape[0]):
    gauss_pars[i, 0] = means[i]
    gauss_pars[i, 1] = variances[i]

moments_pre_analytic = GMN.moments_pre_act_combined_general(gm_pars, gauss_pars, order=5)

print("Comparison of Pre-Activation Moments:")
for i in range(len(moments_pre_samples)):
    print(f"Moment {i+1}:")
    print(f"  Samples: {moments_pre_samples[i]:.4f}")
    print(f"  Analytic: {moments_pre_analytic[i]:.4f}")
    print(f"  Relative Error: {100 * abs((moments_pre_analytic[i] - moments_pre_samples[i]) / moments_pre_samples[i]):.4f} %")
    print()
