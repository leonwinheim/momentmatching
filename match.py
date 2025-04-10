############################################################################################
# This script should demonstrate different Approaches to Moment matching in a NN
# Author: Leon Winheim
# Date: 10.04.2025
############################################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#******Define Functions******
def relu(x):
    # return ReLU
    return np.maximum(0, x)
    alpha = 0.01  # Slope for negative values
    #return np.where(x > 0, x, alpha * x)

def net_func(z_0, w_0, w_1):
    # Compute the elementwise output of the whole network
    return relu(w_1*relu(z_0*w_0))

def net_func_single(z,w):
    # Compute the Elementwise output of the first layer of the network
    return relu(w*z)

#******Computation******
num_samples = 100000

# Generate Input Samples from a Gaussian Distribution
z_0_mean = 5
z_0_var = 1
z_0_samples = np.random.normal(z_0_mean, np.sqrt(z_0_var), num_samples)

# Generate weight samples from a multivariate Gaussian Distribution with definable covariance
mean_vector = [-0.2, 0.2]                        # Mean for w_0 and w_1
cov_matrix = [[1, 0.00], [0.00, 1]]     # Covariance matrix for w_0 and w_1

w_samples = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)
w_0_samples = w_samples[:, 0]  # Extract w_0 samples
w_1_samples = w_samples[:, 1]  # Extract w_1 samples

# Compute the output of the first layer of the network (sample based)
z_1 = net_func_single(z_0_samples, w_0_samples)
z_1_mean = np.mean(z_1)
z_1_var = np.var(z_1)

# Compute the direct sampled output of the whole network (propagate through Network Function)
z_2_direct = net_func(z_0_samples, w_0_samples, w_1_samples)
z_2_mean = np.mean(z_2_direct)
z_2_var = np.var(z_2_direct)

#Generate Samples from the first two moments of the intermediate
z_1_gaussian = np.random.normal(z_1_mean, np.sqrt(z_1_var), num_samples)

# Compute the output of the second layer of the network using the intermediate samples
z_2_intermed = net_func_single(z_1_gaussian, w_1_samples)
z_2_intermed_mean = np.mean(z_2_intermed)
z_2_intermed_var = np.var(z_2_intermed)

# Compare the moments of z_2_direct and z_2_intermed
print("Comparison of Moments:")
print(f"z_2_direct Mean: {z_2_mean:.4f}, Variance: {z_2_var:.4f}")
print(f"z_2_intermed Mean: {z_2_intermed_mean:.4f}, Variance: {z_2_intermed_var:.4f}")

#******Graphics******

# Create a figure with a custom grid layout
fig = plt.figure(figsize=(16, 9))
grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.3)

# First tile: KDEs of input and weight samples
bw_factor = 1
bw_factor_two = 0.1
ax1 = fig.add_subplot(grid[0, 0])
sns.kdeplot(z_0_samples, label="z_0 Samples", color="green", bw_adjust=bw_factor, fill=True, alpha=0.5, ax=ax1)
sns.kdeplot(w_0_samples, label="w_0 Samples", color="blue", bw_adjust=bw_factor, fill=True, alpha=0.5, ax=ax1)
sns.kdeplot(w_1_samples, label="w_1 Samples", color="orange", bw_adjust=bw_factor, fill=True, alpha=0.5, ax=ax1)
ax1.set_title("KDE of Input and Weight Samples")
ax1.set_xlabel("Value")
ax1.set_ylabel("Density")
ax1.legend()

# Second tile: KDEs of z_1 and z_1 Gaussian
ax2 = fig.add_subplot(grid[0, 1])
sns.kdeplot(z_1, label="z_1 Samples", color="green", bw_adjust=bw_factor_two, fill=True, alpha=0.5, ax=ax2)
sns.kdeplot(z_1_gaussian, label="z_1 Gaussian", color="purple", bw_adjust=bw_factor, fill=True, alpha=0.5, ax=ax2)
ax2.set_title("KDE of z_1 and z_1 Gaussian")
ax2.set_xlabel("Value")
ax2.set_ylabel("Density")
ax2.legend()

# Third tile: KDEs of z_2 direct and intermediate samples (spanning two columns)
ax3 = fig.add_subplot(grid[1, :])
sns.kdeplot(z_2_direct, label=f"Direct Samples (mean={z_2_mean:.2f}, var={z_2_var:.2f})", color="blue", bw_adjust=bw_factor_two, fill=True, alpha=0.5, ax=ax3)
sns.kdeplot(z_2_intermed, label=f"Intermediate Samples (mean={z_2_intermed_mean:.2f}, var={z_2_intermed_var:.2f})", color="orange", bw_adjust=bw_factor_two, fill=True, alpha=0.5, ax=ax3)
ax3.set_xlim(left=None, right=5 * z_2_direct.mean())
ax3.set_title("KDE of z_2 Samples")
ax3.set_xlabel("Value")
ax3.set_ylabel("Density")
ax3.legend()

# Add a suptitle and apply tight layout
fig.suptitle("Moment Matching in Neural Networks", fontsize=16,  fontweight='bold')
plt.tight_layout()

# #******Graphics******

# # Create a figure with a custom grid layout
# fig = plt.figure(figsize=(16, 9))
# grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.3)

# # First tile: Histograms of input and weight samples
# bins = 50
# ax1 = fig.add_subplot(grid[0, 0])
# ax1.hist(z_0_samples, bins=bins, label="z_0 Samples", color="green", alpha=0.5, density=True)
# ax1.hist(w_0_samples, bins=bins, label="w_0 Samples", color="blue", alpha=0.5, density=True)
# ax1.hist(w_1_samples, bins=bins, label="w_1 Samples", color="orange", alpha=0.5, density=True)
# ax1.set_title("Histogram of Input and Weight Samples")
# ax1.set_xlabel("Value")
# ax1.set_ylabel("Density")
# ax1.legend()

# # Second tile: Histograms of z_1 and z_1 Gaussian
# ax2 = fig.add_subplot(grid[0, 1])
# ax2.hist(z_1, bins=bins, label="z_1 Samples", color="green", alpha=0.5, density=True)
# ax2.hist(z_1_gaussian, bins=bins, label="z_1 Gaussian", color="purple", alpha=0.5, density=True)
# ax2.set_title("Histogram of z_1 and z_1 Gaussian")
# ax2.set_xlabel("Value")
# ax2.set_ylabel("Density")
# ax2.legend()

# # Third tile: Histograms of z_2 direct and intermediate samples (spanning two columns)
# ax3 = fig.add_subplot(grid[1, :])
# bin_width = 0.01  # Specify the bin width
# bins = np.arange(min(np.min(z_2_direct), np.min(z_2_intermed)), 
#                  max(np.max(z_2_direct), np.max(z_2_intermed)) + bin_width, 
#                  bin_width)
# ax3.hist(z_2_direct, bins=bins, label="Direct Samples", color="blue", alpha=0.5, density=True)
# ax3.hist(z_2_intermed, bins=bins, label="Intermediate Samples", color="orange", alpha=0.5, density=True)
# ax3.set_xlim(min(np.min(z_2_direct), np.min(z_2_intermed)), 0.05 * max(np.max(z_2_direct), np.max(z_2_intermed)))
# ax3.set_title("Histogram of z_2 Samples")
# ax3.set_xlabel("Value")
# ax3.set_ylabel("Density")
# ax3.legend()

# # Add a suptitle and apply tight layout
# fig.suptitle("Moment Matching in Neural Networks", fontsize=16, fontweight='bold')
# plt.tight_layout()

# # Show the plot
# plt.show()
