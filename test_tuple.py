import numpy as np
import matplotlib.pyplot as plt
import GaussianMixtureNetwork as GMN

layers = [1,1,1]
act_func = ['relu','linear']

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