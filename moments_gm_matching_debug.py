import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sklearn
import gm_funcs
import os
import pickle as pkl

#******Define testing connditions******
np.random.seed(42)  # Set seed for reproducibility
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # Limit the number of threads used by joblib, windows 11 stuff

# Define leaky ReLU function
def leaky_relu(x, a):
    return np.where(x > 0, x, a * x)

#################################################################################
#################################################################################
#################################################################################
#******Sample Based Moment Computation******
#Load samples
samples_ext = pkl.load(open("samples_post.pkl", "rb"))

# Parameters for the prior Gaussian Mixture with two components
# a is the slope of the leaky part or the ReLU, c is the variance of the Gaussians, mu are the means of the GM, w are the weights of the GLM
a_test = 1.0
c0_test = 0.2
c1_test = 1.2
mu0_test = 2.0
mu1_test = 5.0
w0_test = 0.2
w1_test = 0.8

#Generate random samples from a Gaussian Mixture
num_samples = 100000
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[mu0_test], [mu1_test]])
gm.covariances_ = np.array([[[c0_test]], [[c1_test]]])
gm.weights_ = np.array([w0_test, w1_test])
samples_prior  = gm.sample(num_samples)[0]

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

# Compute the sample moments oft ext
# Compute the first ten empirical moments of samples_ext
moments_ext = [np.mean(samples_ext**i) for i in range(1, 11)]

print("Comparison of moments")
print("Theoretical moments:")
print("t1:", t1, ", t2:", t2, ", t3:", t3, ", t4:", t4, ", t5:", t5, ", t6:", t6, ", t7:", t7, ", t8:", t8, ", t9:", t9, ", t10:", t10)
print("Sample moments:")
print("t1:", moments_ext[0], ", t2:", moments_ext[1], ", t3:", moments_ext[2], ", t4:", moments_ext[3], ", t5:", moments_ext[4], ", t6:", moments_ext[5], ", t7:", moments_ext[6], ", t8:", moments_ext[7], ", t9:", moments_ext[8], ", t10:", moments_ext[9])
print("Difference between theoretical and sample moments:")
print("t1:", t1 - moments_ext[0], ", t2:", t2 - moments_ext[1], ", t3:", t3 - moments_ext[2], ", t4:", t4 - moments_ext[3], ", t5:", t5 - moments_ext[4], ", t6:", t6 - moments_ext[5], ", t7:", t7 - moments_ext[6], ", t8:", t8 - moments_ext[7], ", t9:", t9 - moments_ext[8], ", t10:", t10 - moments_ext[9])

# FIt the moments
# Initial guess for x, mu, c for optimization
mu00 = 1.0
mu11 = 2.0
w00 = 0.5
w11 = 0.5
c0 = 1.0
c1 = 1.0

#Assemble initial parameters and arguments for the optimizer
params0 = [w00, mu00, mu11, c0, c1]
#args=(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
args = (moments_ext[0], moments_ext[1], moments_ext[2], moments_ext[3], moments_ext[4], moments_ext[5], moments_ext[6], moments_ext[7], moments_ext[8], moments_ext[9])

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

# Ensure seaborn styles are applied
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
plt.title("GM")
sns.kdeplot(samples_prior.squeeze(), label='GM', color='blue', fill=True, alpha=0.5)
sns.kdeplot(samples_prior_fitted.squeeze(), label='GM fitted', color='red', fill=True, alpha=0.5)
sns.kdeplot(samples_ext.squeeze(), label='Samples', color='green', fill=True, alpha=0.5)
plt.legend()

plt.figure(figsize=(10, 6))
plt.title("GM")
plt.hist(samples_prior, bins=500, density=True, label='GM', color='blue', alpha=0.5)
plt.hist(samples_prior_fitted, bins=50, density=True, label='GM fitted', color='red', alpha=0.5)
plt.hist(samples_ext, bins=500, density=True, label='Samples', color='green', alpha=0.5)
plt.legend()