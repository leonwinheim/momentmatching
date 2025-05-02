#################################################################################
# This script tests the analytical moment computation of a GM distribution
# propagated through a leaky ReLU function. Moment Expressions are obtained from
# Mathematica and parsed using sympy.
# Here, we perform the actual Moment matching for two component full mixture
# Author: Leon Winheim
# Date: 29.04.2025
#################################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn
import gm_funcs

#******Define testing connditions******
np.random.seed(42)  # Set seed for reproducibility

# Define leaky ReLU function
def leaky_relu(x, a):
    return np.where(x > 0, x, a * x)

#################################################################################
#################################################################################
#################################################################################
#******Sample Based Moment Computation******
# Parameters for the prior Gaussian Mixture with two components
# a is the slope of the leaky part or the ReLU, c is the variance of the Gaussians, mu are the means of the GLM, w are the weights of the GLM
a_test = 0.05
c0_test = 0.8
c1_test = 1.2
mu0_test = 0.0
mu1_test = 3.0
w0_test = 0.2
w1_test = 0.8

#Generate random samples from a Gaussian Mixture
num_samples = 100000
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[mu0_test], [mu1_test]])
gm.covariances_ = np.array([[[c0_test]], [[c1_test]]])
gm.weights_ = np.array([w0_test, w1_test])
samples_prior  = gm.sample(num_samples)[0]

#Compute propagated samples through leaky ReLU
samples_post = leaky_relu(samples_prior, a_test)

# Compute the first moments empirically
moments_samples = []
for i in range(1, 6):
    moment = np.mean(samples_post ** i)
    moments_samples.append(moment)

# Compute the first Moments analytically
moments_analytic = gm_funcs.compute_moments_analytic(a_test, c0_test, c1_test, mu0_test, mu1_test, w0_test, w1_test)

#******Print the result comparison******
print("*****Moment Comparison*****")
for i in range(1, 6):
    print(f"Order {i}, Samples: {moments_samples[i-1]:.4f}, Analytic: {moments_analytic[i-1]:.4f}")
    print(f"Rel. Error: {abs(moments_samples[i-1] - moments_analytic[i-1])/abs(moments_samples[i-1]) * 100:.2f}%")
    print("")

#################################################################################
#################################################################################
#################################################################################
#******Fit the Initial-Moments******
# Compute the first moments of a Gaussian Mixture with specified parameters
t1 = gm_funcs.e1_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t2 = gm_funcs.e2_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t3 = gm_funcs.e3_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t4 = gm_funcs.e4_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t5 = gm_funcs.e5_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t6 = gm_funcs.e6_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t7 = gm_funcs.e7_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t8 = gm_funcs.e8_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t9 = gm_funcs.e9_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)
t10 = gm_funcs.e10_gm(w0_test,w1_test,mu0_test,mu1_test,c0_test,c1_test)

# Initial guess for x, mu, c for optimization
mu00 = 1.0
mu11 = 2.0
w00 = 0.5
w11 = 0.5
c0 = 1.0
c1 = 1.0

#Assemble initial parameters and arguments for the optimizer
params0 = [w00, mu00, mu11, c0, c1]
args=(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)

# Call moment based optimizer
start = time.time()
result = gm_funcs.fit_gm_moments(params0, args)
end = time.time()
print("Optimization Time Moments:", end - start, "seconds")

print("Optimized parameters Moment-based:")
print("w0:", result.x[0], ", true w0:", w0_test )
print("mu0:", result.x[1], ", true mu0:", mu0_test)
print("mu1:", result.x[2], ", true mu1:", mu1_test)
print("c0:", result.x[3], ", true c0:", c0_test)
print("c1:", result.x[4], ", true c1:", c1_test)
print("Moment Residuals:", result.fun)

# Generate GM Samples with optimized parameters for the moment based method
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[result.x[1]], [result.x[2]]])
gm.covariances_ = np.array([[[result.x[3]]], [[result.x[4]]]])
gm.weights_ = np.array([result.x[0], 1 - result.x[0]])
samples_prior_fitted  = gm.sample(100000)[0]

# Call EM based optimizer on samples
start = time.time()
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.fit(samples_prior)
end = time.time()
print("EM Optimization Time:", end - start, "seconds")
# Extract parameters from the EM-fitted Gaussian Mixture Model
em_weights = gm.weights_
em_means = gm.means_.flatten()
em_covariances = gm.covariances_.flatten()
# Generate GM Samples with optimized parameters for the EM based method
samples_prior_fitted_em = gm.sample(100000)[0]

print("Optimized parameters EM-based:")
print("w0:", em_weights[0], ", true w0:", w0_test)
print("mu0:", em_means[0], ", true mu0:", mu0_test)
print("mu1:", em_means[1], ", true mu1:", mu1_test)
print("c0:", em_covariances[0], ", true c0:", c0_test)
print("c1:", em_covariances[1], ", true c1:", c1_test)

# Ensure seaborn styles are applied
sns.set_theme(style="whitegrid")

# Plot posterior samples as a KDE with filled areas and different colors
plt.figure(figsize=(8, 5))
sns.kdeplot(samples_prior.squeeze(), color='red', label='True Prior', fill=True, alpha=0.5)
sns.kdeplot(samples_prior_fitted.squeeze(), color='green', label='Fitted Pior', fill=True, alpha=0.5)
sns.kdeplot(samples_prior_fitted_em.squeeze(), color='blue', label='Fitted Prior EM', fill=True, alpha=0.5)
plt.title('Prior Samples Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig('workbench/momentmatching_preReLU.png', dpi=300)

#################################################################################
#################################################################################
#################################################################################
#******Fit the Relu-Propagated Samples******
t1 = moments_analytic[0]
t2 = moments_analytic[1]
t3 = moments_analytic[2]
t4 = moments_analytic[3]
t5 = moments_analytic[4]
t6 = moments_analytic[5]
t7 = moments_analytic[6]
t8 = moments_analytic[7]
t9 = moments_analytic[8]
t10 = moments_analytic[9]

# Initial guess for x, mu, c
mu00 = 0.2
mu11 = 2.0
w00 = 0.5
w11 = 0.5
c0 = 0.1
c1 = 1.0

params0 = [w00, mu00, mu11, c0, c1]
args=(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)

# Call moment based optimizer 
start = time.time()
result = gm_funcs.fit_gm_moments(params0, args)
end = time.time()
print("Optimization Time:", end - start, "seconds")

print("Optimized parameters Moment-based:")
print("w0:", result.x[0])
print("mu0:", result.x[1])
print("mu1:", result.x[2])
print("c0:", result.x[3])
print("c1:", result.x[4])
print("Residuals:", result.fun)

# Generate GM Samples with optimized parameters
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[result.x[1]], [result.x[2]]])
gm.covariances_ = np.array([[[result.x[3]]], [[result.x[4]]]])
gm.weights_ = np.array([result.x[0], 1 - result.x[0]])
samples_post_fitted  = gm.sample(100000)[0]

# Call EM based optimizer on samples
start = time.time()
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.fit(samples_post)
end = time.time()
print("EM Optimization Time:", end - start, "seconds")
samples_post_fitted_EM = gm.sample(100000)[0]
# Extract parameters from the EM-fitted Gaussian Mixture Model
em_weights = gm.weights_
em_means = gm.means_.flatten()
em_covariances = gm.covariances_.flatten()

#Print the analyic moments, fitted empirical moments of Moment-based and of EM
print("*****Moment Comparison*****")
print(f"Order {1}, Analytic: {moments_analytic[0]:.4f}, Moment-Based: {gm_funcs.e1_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e1_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {2}, Analytic: {moments_analytic[1]:.4f}, Moment-Based: {gm_funcs.e2_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e2_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {3}, Analytic: {moments_analytic[2]:.4f}, Moment-Based: {gm_funcs.e3_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e3_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {4}, Analytic: {moments_analytic[3]:.4f}, Moment-Based: {gm_funcs.e4_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e4_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {5}, Analytic: {moments_analytic[4]:.4f}, Moment-Based: {gm_funcs.e5_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e5_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {6}, Analytic: {moments_analytic[5]:.4f}, Moment-Based: {gm_funcs.e6_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e6_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {7}, Analytic: {moments_analytic[6]:.4f}, Moment-Based: {gm_funcs.e7_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e7_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {8}, Analytic: {moments_analytic[7]:.4f}, Moment-Based: {gm_funcs.e8_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e8_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {9}, Analytic: {moments_analytic[8]:.4f}, Moment-Based: {gm_funcs.e9_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e9_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")
print(f"Order {10}, Analytic: {moments_analytic[9]:.4f}, Moment-Based: {gm_funcs.e10_gm(result.x[0],1-result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]):.4f}, EM: {gm_funcs.e10_gm(em_weights[0], em_weights[1], em_means[0], em_means[1],em_covariances[0],em_covariances[1]):.4f}")

# Ensure seaborn styles are applied
sns.set_theme(style="whitegrid")

# Plot posterior samples as a KDE with filled areas and different colors
plt.figure(figsize=(8, 5))
sns.kdeplot(samples_post.squeeze(), color='red', label='True Posterior', fill=True, alpha=0.5)
sns.kdeplot(samples_post_fitted.squeeze(), color='green', label='Fitted Posterior', fill=True, alpha=0.5)
sns.kdeplot(samples_post_fitted_EM.squeeze(), color='blue', label='Fitted Posterior EM', fill=True, alpha=0.5)
plt.title('Posterior Samples Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig('workbench/momentmatching_postReLU.png', dpi=300)
