#################################################################################
# This script tests the analytical moment computation of a GM distribution
# propagated through a leaky ReLU function. Moment Expressions are obtained from
# Mathematica and parsed using sympy.
# Here, we perform the actual Moment matching for two component location mixture
# Author: Leon Winheim
# Date: 29.04.2025
#################################################################################
import scipy.stats
from sympy.parsing.mathematica import parse_mathematica
from sympy import var, erf, erfc
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import time
import sklearn
import re
from scipy.optimize import least_squares

#******Define testing connditions******
np.random.seed(42)  # Set seed for reproducibility

# Parameters for the prior Gaussian Location Mixture with two components
# a is the slope of the leaky part or the ReLU, c is the variance of the Gaussians, mu are the means of the GLM, w are the weights of the GLM
a_test = 0.05
c_test = 1
mu0_test = 0.0
mu1_test = 3.0
w0_test = 0.2
w1_test = 0.8

#******Sample Based Moment Computation******
#Generate random samples from a Gaussian Mixture 
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[mu0_test], [mu1_test]])
gm.covariances_ = np.array([[[c_test]], [[c_test]]])
gm.weights_ = np.array([w0_test, w1_test])
samples_prior  = gm.sample(10000)[0]

# Define leaky ReLU function
def leaky_relu(x, a):
    return np.where(x > 0, x, a * x)

#Compute propagated samples through leaky ReLU
samples_post = leaky_relu(samples_prior, a_test)

# Compute the first five moments of the samples
moments_samples = []
for i in range(1, 6):
    moment = np.mean(samples_post ** i)
    moments_samples.append(moment)

# # Plot posterior samples as a kde
# plt.figure(figsize=(10, 6))
# sns.kdeplot(samples_post, fill=True, color='g', label='Posterior Samples')
# plt.title('Posterior Samples Distribution')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.legend()
# plt.grid()

#******Do the analytical moment computation******
# Define the symbolic variables (exclude erf and erfc here)
# a is the slope of the leaky part or the ReLU, c is the variance of the Gaussians, mu are the means of the GLM, w are the weights of the GLM
a, c, mu0, mu1, w0 ,w1 = var('a c mu0 mu1 w0 w1')

# Gaussian Mixture Moments through leaky relu.Expression from mathematica (\[]-Style expressions need to be replaced without brackets)
m = []
# #These are the formulas for 3 components from mathematica
# m1 = r'1/2 (mu[0] w[0] + Erf[mu[0]/(Sqrt[2] c)] mu[0] w[0] + a Erfc[mu[0]/(Sqrt[2] c)] mu[0] w[0] - a c E^(-(mu[1]^2/(2 c^2))) Sqrt[2/\[Pi]] w[1] + mu[1] w[1] + Erf[mu[1]/(Sqrt[2] c)] mu[1] w[1] + a Erfc[mu[1]/(Sqrt[2] c)] mu[1] w[1] + c Sqrt[2/\[Pi]] (-((-1 + a) E^(-(mu[0]^2/(2 c^2))) w[0]) + E^(-(mu[1]^2/(2 c^2))) w[1]) + c E^(-(mu[2]^2/(2 c^2))) Sqrt[2/\[Pi]] w[2] - a c E^(-(mu[2]^2/(2 c^2))) Sqrt[2/\[Pi]] w[2] + mu[2] w[2] + Erf[mu[2]/(Sqrt[2] c)] mu[2] w[2] + a Erfc[mu[2]/(Sqrt[2] c)] mu[2] w[2])' 
# m2 = r'1/2 Erf[mu[0]/(Sqrt[2] c)] (c^2 + mu[0]^2) w[0] + 1/2 (-((-1 + a^2) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]] mu[0] w[0]) + mu[0]^2 w[0] + a^2 Erfc[mu[0]/(Sqrt[2] c)] (c^2 + mu[0]^2) w[0] + c^2 (w[0] + w[1]) + E^(-((mu[1]^2 + mu[2]^2)/(2 c^2))) (-((-1 + a^2) c E^(mu[1]^2/(2 c^2)) Sqrt[2/\[Pi]]mu[2] w[2]) + E^(mu[2]^2/(2 c^2)) (-((-1 + a^2) c Sqrt[2/\[Pi]] mu[1] w[1]) + E^(mu[1]^2/(2 c^2)) ((mu[1]^2 + (Erf[mu[1]/(Sqrt[2] c)] + a^2 Erfc[mu[1]/(Sqrt[2] c)]) (c^2 + mu[1]^2)) w[1] + (1 + Erf[mu[2]/(Sqrt[2] c)] + a^2 Erfc[mu[2]/(Sqrt[2] c)]) (c^2 + mu[2]^2) w[2]))))'
# m3 = r'((c E^(-(mu[0]^2/(2 c^2))) (2 c^2 + mu[0]^2))/Sqrt[2 \[Pi]] + 1/2 (1 + Erf[mu[0]/(Sqrt[2] c)]) mu[0] (3 c^2 + mu[0]^2)) w[0] + (-((a^3 c E^(-(mu[0]^2/(2 c^2))) (2 c^2 + mu[0]^2))/Sqrt[2 \[Pi]]) + 1/2 a^3 Erfc[mu[0]/(Sqrt[2] c)] mu[0] (3 c^2 + mu[0]^2)) w[0] + ((c E^(-(mu[1]^2/(2 c^2))) (2 c^2 + mu[1]^2))/Sqrt[2 \[Pi]] + 1/2 (1 + Erf[mu[1]/(Sqrt[2] c)]) mu[1] (3 c^2 + mu[1]^2)) w[1] + (-((a^3 c E^(-(mu[1]^2/(2 c^2))) (2 c^2 + mu[1]^2))/Sqrt[2 \[Pi]]) + 1/2 a^3 Erfc[mu[1]/(Sqrt[2] c)] mu[1] (3 c^2 + mu[1]^2)) w[1] + ((c E^(-(mu[2]^2/(2 c^2))) (2 c^2 + mu[2]^2))/Sqrt[2 \[Pi]] + 1/2 (1 + Erf[mu[2]/(Sqrt[2] c)]) mu[2] (3 c^2 + mu[2]^2)) w[2] + (-((a^3 c E^(-(mu[2]^2/(2 c^2))) (2 c^2 + mu[2]^2))/Sqrt[2 \[Pi]]) + 1/2 a^3 Erfc[mu[2]/(Sqrt[2] c)] mu[2] (3 c^2 + mu[2]^2)) w[2]'
# m4 = r'1/2 Erf[mu[0]/(Sqrt[2] c)] (3 c^4 + 6 c^2 mu[0]^2 + mu[0]^4) w[0] + 1/2 (-((-1 + a^4) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]]mu[0] (5 c^2 + mu[0]^2) w[0]) + (3 c^4 + 6 c^2 mu[0]^2 + mu[0]^4) w[0] + a^4 Erfc[mu[0]/(Sqrt[2] c)] (3 c^4 + 6 c^2 mu[0]^2 + mu[0]^4) w[0] + 6 c^4 w[1] + E^(-((mu[1]^2 + mu[2]^2)/(2 c^2))) (-((-1 + a^4) c E^(mu[1]^2/(2 c^2)) Sqrt[2/\[Pi]] mu[2] (5 c^2 + mu[2]^2) w[2]) + E^(mu[2]^2/(2 c^2)) (-((-1 + a^4) c Sqrt[2/\[Pi]] mu[1] (5 c^2 + mu[1]^2) w[1]) + E^(mu[1]^2/(2 c^2)) (2 mu[1]^2 (6 c^2 + mu[1]^2) w[1] + (-1 + a^4) Erfc[mu[1]/(Sqrt[2] c)] (3 c^4 + 6 c^2 mu[1]^2 + mu[1]^4) w[1] + (2 + (-1 + a^4) Erfc[mu[2]/(Sqrt[2] c)]) (3 c^4 + 6 c^2 mu[2]^2 + mu[2]^4) w[2]))))'
# m5 = r'1/2 (10 c^2 mu[0]^3 w[0] + mu[0]^5 w[0] + c^4 (-8 (-1 + a^5) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]] + 15 mu[0]) w[0] - (-1 + a^5) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]] mu[0]^2 (9 c^2 + mu[0]^2) w[0] + Erf[mu[0]/(Sqrt[2] c)] mu[0] (15 c^4 + 10 c^2 mu[0]^2 + mu[0]^4) w[0] + a^5 Erfc[mu[0]/(Sqrt[2] c)] mu[0] (15 c^4 + 10 c^2 mu[0]^2 + mu[0]^4) w[0] + 30 c^4 mu[1] w[1] + 20 c^2 mu[1]^3 w[1] + 2 mu[1]^5 w[1] - (-1 + a^5) c E^(-(mu[1]^2/(2 c^2))) Sqrt[2/\[Pi]] (c^2 + mu[1]^2) (8 c^2 + mu[1]^2) w[1] + (-1 + a^5) Erfc[mu[1]/(Sqrt[2] c)] mu[1] (15 c^4 + 10 c^2 mu[1]^2 + mu[1]^4) w[1] - (-1 + a^5) c E^(-(mu[2]^2/(2 c^2))) Sqrt[2/\[Pi]] (c^2 + mu[2]^2) (8 c^2 + mu[2]^2) w[2] + (2 + (-1 + a^5) Erfc[mu[2]/(Sqrt[2] c)]) mu[2] (15 c^4 + 10 c^2 mu[2]^2 + mu[2]^4) w[2])'

#These are the formulas for 2 components from mathematica
m1 = r'(E^(-((mu[0]^2 + mu[1]^2)/(2 c^2))) (-Sqrt[2] (-1 + a) c E^(mu[1]^2/(2 c^2)) w[0] + E^(mu[0]^2/(2 c^2)) (-Sqrt[2] (-1 + a) c w[1] + E^(mu[1]^2/(2 c^2)) Sqrt[\[Pi]] ((1 + Erf[mu[0]/(Sqrt[2] c)] + a Erfc[mu[0]/(Sqrt[2] c)]) mu[0] w[0] + (1 + Erf[mu[1]/(Sqrt[2] c)] + a Erfc[mu[1]/(Sqrt[2] c)]) mu[1] w[1]))))/(2 Sqrt[\[Pi]])'
m2 = r'1/2 Erf[mu[0]/(Sqrt[2] c)] (c^2 + mu[0]^2) w[0] + (E^(-((mu[0]^2 + mu[1]^2)/(2 c^2))) (-Sqrt[2] (-1 + a^2) c E^(mu[0]^2/(2 c^2)) mu[1] w[1] + E^(mu[1]^2/(2 c^2)) (-Sqrt[2] (-1 + a^2) c mu[0] w[0] + E^(mu[0]^2/(2 c^2)) Sqrt[\[Pi]] ((1 + a^2 Erfc[mu[0]/(Sqrt[2] c)]) (c^2 + mu[0]^2) w[0] + (1 + Erf[mu[1]/(Sqrt[2] c)] + a^2 Erfc[mu[1]/(Sqrt[2] c)]) (c^2 + mu[1]^2) w[1]))))/(2 Sqrt[\[Pi]])'
m3 = r'1/2 Erf[mu[0]/(Sqrt[2] c)] mu[0] (3 c^2 + mu[0]^2) w[0] + 1/(2 Sqrt[\[Pi]]) E^(-((mu[0]^2 + mu[1]^2)/(2 c^2))) (-Sqrt[2] (-1 + a^3) c E^(mu[1]^2/(2 c^2)) (2 c^2 + mu[0]^2) w[0] + E^(mu[0]^2/(2 c^2)) (-Sqrt[2] (-1 + a^3) c (2 c^2 + mu[1]^2) w[1] + E^(mu[1]^2/(2 c^2)) Sqrt[\[Pi]] ((1 + a^3 Erfc[mu[0]/(Sqrt[2] c)]) mu[0] (3 c^2 + mu[0]^2) w[0] + (1 + Erf[mu[1]/(Sqrt[2] c)] + a^3 Erfc[mu[1]/(Sqrt[2] c)]) mu[1] (3 c^2 + mu[1]^2) w[1])))'
m4 = r'3 c^4 w[0] + 1/(2 Sqrt[\[Pi]]) E^(-((mu[0]^2 + mu[1]^2)/(2 c^2))) (-Sqrt[2] (-1 + a^4) c E^(mu[0]^2/(2 c^2)) mu[1] (5 c^2 + mu[1]^2) w[1] + E^(mu[1]^2/(2 c^2)) (-Sqrt[2] (-1 + a^4) c mu[0] (5 c^2 + mu[0]^2) w[0] + E^(mu[0]^2/(2 c^2)) Sqrt[\[Pi]] (2 mu[0]^2 (6 c^2 + mu[0]^2) w[0] + (-1 + a^4) Erfc[mu[0]/(Sqrt[2] c)] (3 c^4 + 6 c^2 mu[0]^2 + mu[0]^4) w[0] + (2 + (-1 + a^4) Erfc[mu[1]/(Sqrt[2] c)]) (3 c^4 + 6 c^2 mu[1]^2 + mu[1]^4) w[1])))'
m5 = r'1/2 Erf[mu[0]/(Sqrt[2] c)] mu[0] (15 c^4 + 10 c^2 mu[0]^2 + mu[0]^4) w[0] + 1/(2 Sqrt[\[Pi]]) E^(-((mu[0]^2 + mu[1]^2)/(2 c^2))) (-Sqrt[2] (-1 + a^5) c E^(mu[1]^2/(2 c^2)) (c^2 + mu[0]^2) (8 c^2 + mu[0]^2) w[0] + E^(mu[0]^2/(2 c^2)) (-Sqrt[2] (-1 + a^5) c (c^2 + mu[1]^2) (8 c^2 + mu[1]^2) w[1] + E^(mu[1]^2/(2 c^2)) Sqrt[\[Pi]] ((1 + a^5 Erfc[mu[0]/(Sqrt[2] c)]) mu[0] (15 c^4 + 10 c^2 mu[0]^2 + mu[0]^4) w[0] + (2 + (-1 + a^5) Erfc[mu[1]/(Sqrt[2] c)]) mu[1] (15 c^4 + 10 c^2 mu[1]^2 + mu[1]^4) w[1])))'

# Replace \[Pi] with Pi in the expressions so python can read it
def replace_pi(expression):
    return expression.replace("\\[Pi]", "Pi")

# Replace mu[i] with mui when i is a variable integer
def replace_mu_index(expression):
    # Use regex to find patterns like mu[i] where i is an integer
    pattern = r"mu\[(\d+)\]"
    # Replace mu[i] with mui
    return re.sub(pattern, lambda match: f"mu{match.group(1)}", expression)

# Replace w[i] with wi when i is a variable integer
def replace_w_index(expression):
    # Use regex to find patterns like w[i] where i is an integer
    pattern = r"w\[(\d+)\]"
    # Replace w[i] with wi
    return re.sub(pattern, lambda match: f"w{match.group(1)}", expression)

# Apply the replacement to all moment expressions
m1, m2, m3, m4, m5 = map(replace_w_index, [m1, m2, m3, m4, m5])

# Apply the replacement to all moment expressions
m1, m2, m3, m4, m5 = map(replace_mu_index, [m1, m2, m3, m4, m5])
    
# Replace \[Pi] with Pi in all moment expressions
m1, m2, m3, m4, m5 = map(replace_pi, [m1, m2, m3, m4, m5])

m.append(m1)
m.append(m2)
m.append(m3)
m.append(m4)
m.append(m5)

# # Print the processed moment expressions
# for i, moment in enumerate(m, start=1):
#     print(f"Moment {i}: {moment}")

#Parse every expression for translation
mp = []
for mx in m:
    mp.append(parse_mathematica(mx)) 

#Replace the function expressions with handles
for i, mx in enumerate(mp):
    mp[i] = mx.replace(
        lambda x: x.func.__name__ == 'Erf',
        lambda x: erf(x.args[0])
    ).replace(
        lambda x: x.func.__name__ == 'Erfc',
        lambda x: erfc(x.args[0])
    )

#Substitute Values
values = {a: a_test, c: c_test, mu0: mu0_test, mu1:mu1_test, w0: w0_test, w1: w1_test}
moments_analytic = []
for i, mx in enumerate(mp):
    evaluated_expr = mx.subs(values)
    # Numerically evaluate the result
    result = evaluated_expr.evalf()
    moments_analytic.append(result)

#******Print the result comparison******
print("*****Moment Comparison*****")
for i in range(1, 6):
    print(f"Order {i}, Samples: {moments_samples[i-1]:.4f}, Analytic: {moments_analytic[i-1]:.4f}")
    print(f"Rel. Error: {abs(moments_samples[i-1] - moments_analytic[i-1])/abs(moments_samples[i-1]) * 100:.2f}%")
    print("")

#******Implement the  Moment-Matching******
# Noncentral moments of a Gaussian
def  e1(mu):
    return mu
def  e2(mu,c_v):
    return c_v + mu**2
def  e3(mu,c_v):
    return 3*c_v*mu+mu**3
def  e4(mu,c_v):
    return mu**4+6*(mu**2)*c_v+3*c_v**2
def  e5(mu,c_v):
    return mu**5+10*(mu**3)*c_v+15*mu*c_v**2

# Noncentral moments of a Gaussian Mixture
def e1_gm(w0_v,w1_v,mu0_v,mu1_v,c_v):
    return (w0_v*e1(mu0_v) + w1_v*e1(mu1_v))

def e2_gm(w0_v,w1_v,mu0_v,mu1_v,c_v):
    return (w0_v*e2(mu0_v,c_v) + w1_v*e2(mu1_v,c_v))

def e3_gm(w0_v,w1_v,mu0_v,mu1_v,c_v):
    return (w0_v*e3(mu0_v,c_v) + w1_v*e3(mu1_v,c_v))

def e4_gm(w0_v,w1_v,mu0_v,mu1_v,c_v):
    return (w0_v*e4(mu0_v,c_v) + w1_v*e4(mu1_v,c_v))

def e5_gm(w0_v,w1_v,mu0_v,mu1_v,c_v):
    return (w0_v*e5(mu0_v,c_v) + w1_v*e5(mu1_v,c_v))

# Function to compute the residuals for optimization
def residuals(params, t1, t2, t3, t4, t5):
    w0_v, mu0_v, mu1_v, c_v = params
    w1_v = 1 - w0_v                                 #Attention! For more components we need to enforce the weights will be 1 in sum differently
    r = np.array([
        e1_gm(w0_v,w1_v,mu0_v,mu1_v,c_v) - t1,
        e2_gm(w0_v,w1_v,mu0_v,mu1_v,c_v) - t2,
        e3_gm(w0_v,w1_v,mu0_v,mu1_v,c_v) - t3,
        e4_gm(w0_v,w1_v,mu0_v,mu1_v,c_v) - t4,
        e5_gm(w0_v,w1_v,mu0_v,mu1_v,c_v) - t5
    ], dtype=float)

    return r

#******Fit the Initial-Moments******
t1 = e1_gm(w0_test,w1_test,mu0_test,mu1_test,c_test)
t2 = e2_gm(w0_test,w1_test,mu0_test,mu1_test,c_test)
t3 = e3_gm(w0_test,w1_test,mu0_test,mu1_test,c_test)
t4 = e4_gm(w0_test,w1_test,mu0_test,mu1_test,c_test)
t5 = e5_gm(w0_test,w1_test,mu0_test,mu1_test,c_test)

# Initial guess for x, mu, c
mu00 = 1.0
mu11 = 2.0
w00 = 0.5
w11 = 0.5
c0 = 1.0

params0 = [w00, mu00, mu11, c0]

# Call optimizer with bounds
result = least_squares(residuals,params0,args=(t1, t2, t3, t4, t5))

print("Optimized parameters:")
print("w0:", result.x[0])
print("mu0:", result.x[1])
print("mu1:", result.x[2])
print("c:", result.x[3])
print("Residuals:", result.fun)

# Generate GM Samples with optimized parameters
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[result.x[1]], [result.x[2]]])
gm.covariances_ = np.array([[[result.x[3]]], [[result.x[3]]]])
gm.weights_ = np.array([result.x[0], 1 - result.x[0]])
samples_prior_fitted  = gm.sample(100000)[0]

# Ensure seaborn styles are applied
sns.set_theme(style="whitegrid")

# Plot posterior samples as a KDE with filled areas and different colors
plt.figure(figsize=(8, 5))
sns.kdeplot(samples_prior, color='red', label='True Posterior', fill=True, alpha=0.5)
sns.kdeplot(samples_prior_fitted, color='green', label='Fitted Posterior', fill=True, alpha=0.5)
plt.title('Posterior Samples Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig('workbench/momentmatching_preReLU.png', dpi=300)

#******Fit the Relu-Proppagated Samples******
t1 = moments_analytic[0]
t2 = moments_analytic[1]
t3 = moments_analytic[2]
t4 = moments_analytic[3]
t5 = moments_analytic[4]

# Initial guess for x, mu, c
mu00 = 1.0
mu11 = 2.0
w00 = 0.5
w11 = 0.5
c0 = 1.0

params0 = [w00, mu00, mu11, c0]

# Call optimizer with bounds
start = time.time()
result = least_squares(residuals,params0,args=(t1, t2, t3, t4, t5))
end = time.time()
print("Optimization Time:", end - start, "seconds")

print("Optimized parameters:")
print("w0:", result.x[0])
print("mu0:", result.x[1])
print("mu1:", result.x[2])
print("c:", result.x[3])
print("Residuals:", result.fun)

# Generate GM Samples with optimized parameters
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[result.x[1]], [result.x[2]]])
gm.covariances_ = np.array([[[result.x[3]]], [[result.x[3]]]])
gm.weights_ = np.array([result.x[0], 1 - result.x[0]])
samples_post_fitted  = gm.sample(100000)[0]

# Ensure seaborn styles are applied
sns.set_theme(style="whitegrid")

# Plot posterior samples as a KDE with filled areas and different colors
plt.figure(figsize=(8, 5))
sns.kdeplot(samples_post, color='red', label='True Posterior', fill=True, alpha=0.5)
sns.kdeplot(samples_post_fitted, color='green', label='Fitted Posterior', fill=True, alpha=0.5)
plt.title('Posterior Samples Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig('workbench/momentmatching_postReLU.png', dpi=300)
