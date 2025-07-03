import numpy as np
import matplotlib.pyplot as plt
import GaussianMixtureNetwork as GMN
from sklearn.mixture import GaussianMixture
import os
import pickle

os.environ["LOKY_MAX_CPU_COUNT"] = "10"

#******Control variables******
#np.random.seed(5)  # For reproducibility

#******Network Architecture******
# layers = [1,50,1]
# act_func = ['relu','linear']
layers = [1,10,10,10,1]
act_func = ['relu','relu','relu','linear']
# layers = [1,5,5,1]
# act_func = ['relu','relu','linear']
gm_comp_pre = 2
gm_comp_post =2
moments_pre = 5
moments_post = 5
a_relu = 0.0

print("Initializing model...")
model = GMN.GaussianMixtureNetwork(layers,act_func,gm_comp_pre,gm_comp_post,moments_pre,moments_post,a_relu,peak=False)
print("Model initialized.")

x = np.array([[2.0]])   # Example input value

#model.compare_sample_moments_forward_special(x) #Perform all forward passes so the moments are stored, KBNN has to be done seperately
model.forward_combined(x)

#******Evaluate the moments*****
# Output
predictive_moments_mm = []
for i in range(1, moments_post + 1):
    predictive_moments_mm.append(GMN.gm_noncentral_moment(i,model.weights_gm_post[-1][0,:],model.means_gm_post[-1][0,:],model.variances_gm_post[-1][0,:]))

predictive_moments_samples = []
for i in range(1, moments_post + 1):
    predictive_moments_samples.append(model.post_activation_moments_samples[-1][0,i-1])

# predictive_moments_em = []
# for i in range(1, moments_post + 1):
#     moment = GMN.gm_noncentral_moment(i, model.weights_gm_post_em[-1][0, :], model.means_gm_post_em[-1][0, :], model.variances_gm_post_em[-1][0, :])
#     predictive_moments_em.append(moment)

# Compute the rel errors
relative_errors_mm_samples = []
relative_errors_em_samples = []
for i in range(moments_post):
    rel_error_mm_samples = 100 * abs((predictive_moments_mm[i] - predictive_moments_samples[i]) / predictive_moments_samples[i])
    relative_errors_mm_samples.append(rel_error_mm_samples)
    
#     rel_error_em_samples = 100 * abs((predictive_moments_em[i] - predictive_moments_samples[i]) / predictive_moments_samples[i])
#     relative_errors_em_samples.append(rel_error_em_samples)

# rel_error_em_samples = np.array(relative_errors_em_samples)
rel_error_mm_samples = np.array(relative_errors_mm_samples)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

#******Do the same for KBNN******
np.random.seed(41)  # Reset seed for reproducibility

#******Network Architecture******
# layers = [1,50,1]
# act_func = ['relu','linear']
layers = [1,10,10,10,1]
act_func = ['relu','relu','relu','linear']
gm_comp_pre = 1
gm_comp_post = 1
moments_pre = 2
moments_post = 2
moments_eval = 5
a_relu = 0.0

# print("Initializing second model...")
# model = GMN.GaussianMixtureNetwork(layers,act_func,gm_comp_pre,gm_comp_post,moments_pre,moments_post,a_relu,peak=False)
# print("Second Model initialized.")

# model.forward_moments_ref(x)  # Perform forward pass to compute moments

# predictive_moments_kbnn = []
# for i in range(1, moments_eval + 1):
#     predictive_moments_kbnn.append(GMN.gm_noncentral_moment(i, model.weights_gm_post[-1][0, :], model.means_gm_post[-1][0, :], model.variances_gm_post[-1][0, :]))

# # Compute the relative errors for KBNN
# relative_errors_kbnn_samples = []
# for i in range(moments_eval):
#     rel_error_kbnn_samples = 100 * abs((predictive_moments_kbnn[i] - predictive_moments_samples[i]) / predictive_moments_samples[i])
#     relative_errors_kbnn_samples.append(rel_error_kbnn_samples)

#******Compare the moments******
print("Comparing moments...")
for i in range(moments_eval):
    print(f"Moment {i+1}:")
    print(f"  Samples: {predictive_moments_samples[i]:.4f}")
    print(f"  Predictive (MM): {predictive_moments_mm[i]:.4f}")
    #print(f"  KBNN: {predictive_moments_kbnn[i]:.4f}")
    #print(f"  EM: {predictive_moments_em[i]:.4f}")
    print(f"  Relative Error (MM vs Samples): {100 * abs((predictive_moments_mm[i] - predictive_moments_samples[i]) / predictive_moments_samples[i]):.4f} %")
    #print(f"  Relative Error (KBNN vs Samples): {100 * abs((predictive_moments_kbnn[i] - predictive_moments_samples[i]) / predictive_moments_samples[i]):.4f} %")
    #print(f"  Relative Error (EM vs Samples): {100 * abs((predictive_moments_em[i] - predictive_moments_samples[i]) / predictive_moments_samples[i]):.4f} %")
    print()

#******Make bar plot to visualize this******

plt.figure(figsize=(12, 6))
moments_post = moments_eval
x_labels = [f'Moment {i+1}' for i in range(moments_post)]
x = np.arange(len(x_labels))  # the label locations
width = 0.18  # the width of the bars

bars_samples = plt.bar(x - 1.5*width, [1.0]*moments_post, width, label='Samples', color='green', alpha=0.6)
bars_mm = plt.bar(x - 0.5*width, 1 + rel_error_mm_samples/100, width, label='Moment Matching', color='blue', alpha=0.6)
#bars_em = plt.bar(x + 0.5*width, 1 + rel_error_em_samples/100, width, label='EM', color='red', alpha=0.6)
#bars_kbnn = plt.bar(x + 1.5*width, 1 + np.array(relative_errors_kbnn_samples)/100, width, label='KBNN', color='orange', alpha=0.6)

# Add relative error as text above each bar (except for samples, which is the reference)
for i in range(moments_post):
    plt.text(x[i] - 0.5*width, 1 + rel_error_mm_samples[i]/100 + 0.02, f"{rel_error_mm_samples[i]:.2f}%", ha='center', va='bottom', fontsize=8, color='blue')
    #plt.text(x[i] + 0.5*width, 1 + rel_error_em_samples[i]/100 + 0.02, f"{rel_error_em_samples[i]:.2f}%", ha='center', va='bottom', fontsize=8, color='red')
    #plt.text(x[i] + 1.5*width, 1 + relative_errors_kbnn_samples[i]/100 + 0.02, f"{relative_errors_kbnn_samples[i]:.2f}%", ha='center', va='bottom', fontsize=8, color='orange')

plt.xlabel('Moments')
plt.ylabel('Value')
plt.title('Comparison of Predictive Moments for Samples, GM-MM, KBNN and EM')
plt.xticks(x, x_labels)
plt.legend()
plt.tight_layout()
plt.savefig('figures/moment_comparison.png')



