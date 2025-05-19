import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import mixture
import gm_funcs

os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # Limit the number of threads used by joblib, windows 11 stuff


# Define parameters for the scalar (1D) Gaussian mixture
n_samples = 100000
means = np.array([[0], [5]])
covariances = [np.array([[1]]), np.array([[1]])]
weights = [0.5, 0.5]

# Generate synthetic data from the defined GMM parameters
n_samples_1 = int(n_samples * weights[0])
n_samples_2 = n_samples - n_samples_1
samples_1 = np.random.normal(loc=means[0, 0], scale=np.sqrt(covariances[0][0, 0]), size=n_samples_1)
samples_2 = np.random.normal(loc=means[1, 0], scale=np.sqrt(covariances[1][0, 0]), size=n_samples_2)
samples = np.concatenate([samples_1, samples_2]).reshape(-1, 1)

# Fit a GaussianMixture model for 1D data
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(samples)

# Generate samples from the fitted GMM
gmm_samples, labels = gmm.sample(n_samples)
gmm_samples = gmm_samples.flatten()

sns.kdeplot(x=gmm_samples, fill=True, color="purple", thresh=0.05)
plt.title("KDE of 1D GMM Samples")
plt.xlabel("X")
plt.ylabel("Density")
plt.show()

# Perform momentmatching
moments = np.zeros((10, ))
for i in range(1, 11):
    moments[i-1] = np.mean(gmm_samples ** i)

pars = gm_funcs.match_moments(moments,2)


print("Pars")
print(pars)