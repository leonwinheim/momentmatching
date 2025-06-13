import pickle
import numpy as np
import GaussianMixtureNetwork as GMN

# Load the data
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data

i = 3
l = 2

z_complete = np.stack((model.means_gm_post[l-1][:, :], model.variances_gm_post[l-1][:, :], model.weights_gm_post[l-1][:, :]),axis=1)
z_complete = np.concatenate((z_complete, np.zeros((1, 3, model.components_post))), axis=0)
z_complete[-1, 0, 0] = 1 # Bias as fictive GM component with mean 1, cov 0 
z_complete[-1, 1, 0] = 0 # and variance 0
z_complete[-1, 2, 0] = 1 # and weight 1

w_complete = np.stack((model.weight_means[l][:,i], model.weight_variances[l][:,i]), axis=1)


moments_pre = GMN.moments_pre_act_combined_general(z_complete, w_complete,order=model.moments_pre)

print(moments_pre)
print(model.pre_activation_moments_samples[l][i,:])

