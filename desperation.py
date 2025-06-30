import pickle
import numpy as np
import GaussianMixtureNetwork as GMN
import matplotlib.pyplot as plt

# Load the data
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data

i = 0
l = 2

# Analytical Moments
z_complete = np.stack((model.means_gm_post[l-1][:, :], model.variances_gm_post[l-1][:, :], model.weights_gm_post[l-1][:, :]),axis=1)
z_complete = np.concatenate((z_complete, np.zeros((1, 3, model.components_post))), axis=0)
z_complete[-1, 0, 0] = 1 # Bias as fictive GM component with mean 1, cov 0 
z_complete[-1, 1, 0] = 0 # and variance 0
z_complete[-1, 2, 0] = 1 # and weight 1

w_complete = np.stack((model.weight_means[l][:,i], model.weight_variances[l][:,i]), axis=1)


moments_pre = GMN.moments_pre_act_combined_general(z_complete, w_complete,order=model.moments_pre)

print(moments_pre)
print(model.pre_activation_moments_samples[l][i,:])

plt.figure(figsize=(10, 6))
plt.title("Pre-Activation Moments")
plt.hist(model.pre_activation_samples[l][:,i], bins=500,density=True, alpha=0.5, label='Sampled Moments')

plt.figure(figsize=(10, 6))
plt.title("Post-Activation Moments")
plt.hist(model.post_activation_samples[l+1][:,i], bins=500,density=True, alpha=0.5, label='Sampled Moments')


print("Post-Activation Moments 0:")
for n in range(5):
    print(model.post_activation_moments_analytic[0][n,:])

print("Post-Activation Moments 1:")
for n in range(5):
    print(model.post_activation_moments_analytic[1][n,:])

print("Post-Activation Moments 2:")
for n in range(5):
    print(model.post_activation_moments_analytic[2][n,:])