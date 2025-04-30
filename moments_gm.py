#################################################################################
# This script tests the analytical moment computation of a GM distribution
# propagated through a leaky ReLU function. Moment Expressions are obtained from
# Mathematica and parsed using sympy.
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

#******Define testing connditions******
np.random.seed(42)  # Set seed for reproducibility

a_test = 0.05
c_test = 1
mu0_test = 0.0
mu1_test = 3.0
w0_test = 0.2
w1_test = 0.8

#******Sample Based Moment Computation******
#Generate random samples from a Gaussian Mixturefrom scikit learn
gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
gm.means_ = np.array([[mu0_test], [mu1_test]])
gm.covariances_ = np.array([[[c_test]], [[c_test]]])
gm.weights_ = np.array([w0_test, w1_test])
samples_prior  = gm.sample(1000000)[0]

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

# Plot posterior samples as a kde
plt.figure(figsize=(10, 6))
sns.kdeplot(samples_post, fill=True, color='g', label='Posterior Samples')
plt.title('Posterior Samples Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()

#******Do the analytical moment computation******
# Define the symbolic variables (exclude erf and erfc here)
# a is the slope of the leaky part or the ReLU, c is the variance of the Gaussian, mu is the mean of the Gaussian
a, c, mu0, mu1, w0 ,w1 = var('a c mu0 mu1 w0 w1')

# Gaussian Mixture Moments through leaky relu.Expression from mathematica (\[]-Style expressions need to be replaced without brackets)
m = []
# #These are the formulas for 3 components
# m1 = r'1/2 (mu[0] w[0] + Erf[mu[0]/(Sqrt[2] c)] mu[0] w[0] + a Erfc[mu[0]/(Sqrt[2] c)] mu[0] w[0] - a c E^(-(mu[1]^2/(2 c^2))) Sqrt[2/\[Pi]] w[1] + mu[1] w[1] + Erf[mu[1]/(Sqrt[2] c)] mu[1] w[1] + a Erfc[mu[1]/(Sqrt[2] c)] mu[1] w[1] + c Sqrt[2/\[Pi]] (-((-1 + a) E^(-(mu[0]^2/(2 c^2))) w[0]) + E^(-(mu[1]^2/(2 c^2))) w[1]) + c E^(-(mu[2]^2/(2 c^2))) Sqrt[2/\[Pi]] w[2] - a c E^(-(mu[2]^2/(2 c^2))) Sqrt[2/\[Pi]] w[2] + mu[2] w[2] + Erf[mu[2]/(Sqrt[2] c)] mu[2] w[2] + a Erfc[mu[2]/(Sqrt[2] c)] mu[2] w[2])' 
# m2 = r'1/2 Erf[mu[0]/(Sqrt[2] c)] (c^2 + mu[0]^2) w[0] + 1/2 (-((-1 + a^2) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]] mu[0] w[0]) + mu[0]^2 w[0] + a^2 Erfc[mu[0]/(Sqrt[2] c)] (c^2 + mu[0]^2) w[0] + c^2 (w[0] + w[1]) + E^(-((mu[1]^2 + mu[2]^2)/(2 c^2))) (-((-1 + a^2) c E^(mu[1]^2/(2 c^2)) Sqrt[2/\[Pi]]mu[2] w[2]) + E^(mu[2]^2/(2 c^2)) (-((-1 + a^2) c Sqrt[2/\[Pi]] mu[1] w[1]) + E^(mu[1]^2/(2 c^2)) ((mu[1]^2 + (Erf[mu[1]/(Sqrt[2] c)] + a^2 Erfc[mu[1]/(Sqrt[2] c)]) (c^2 + mu[1]^2)) w[1] + (1 + Erf[mu[2]/(Sqrt[2] c)] + a^2 Erfc[mu[2]/(Sqrt[2] c)]) (c^2 + mu[2]^2) w[2]))))'
# m3 = r'((c E^(-(mu[0]^2/(2 c^2))) (2 c^2 + mu[0]^2))/Sqrt[2 \[Pi]] + 1/2 (1 + Erf[mu[0]/(Sqrt[2] c)]) mu[0] (3 c^2 + mu[0]^2)) w[0] + (-((a^3 c E^(-(mu[0]^2/(2 c^2))) (2 c^2 + mu[0]^2))/Sqrt[2 \[Pi]]) + 1/2 a^3 Erfc[mu[0]/(Sqrt[2] c)] mu[0] (3 c^2 + mu[0]^2)) w[0] + ((c E^(-(mu[1]^2/(2 c^2))) (2 c^2 + mu[1]^2))/Sqrt[2 \[Pi]] + 1/2 (1 + Erf[mu[1]/(Sqrt[2] c)]) mu[1] (3 c^2 + mu[1]^2)) w[1] + (-((a^3 c E^(-(mu[1]^2/(2 c^2))) (2 c^2 + mu[1]^2))/Sqrt[2 \[Pi]]) + 1/2 a^3 Erfc[mu[1]/(Sqrt[2] c)] mu[1] (3 c^2 + mu[1]^2)) w[1] + ((c E^(-(mu[2]^2/(2 c^2))) (2 c^2 + mu[2]^2))/Sqrt[2 \[Pi]] + 1/2 (1 + Erf[mu[2]/(Sqrt[2] c)]) mu[2] (3 c^2 + mu[2]^2)) w[2] + (-((a^3 c E^(-(mu[2]^2/(2 c^2))) (2 c^2 + mu[2]^2))/Sqrt[2 \[Pi]]) + 1/2 a^3 Erfc[mu[2]/(Sqrt[2] c)] mu[2] (3 c^2 + mu[2]^2)) w[2]'
# m4 = r'1/2 Erf[mu[0]/(Sqrt[2] c)] (3 c^4 + 6 c^2 mu[0]^2 + mu[0]^4) w[0] + 1/2 (-((-1 + a^4) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]]mu[0] (5 c^2 + mu[0]^2) w[0]) + (3 c^4 + 6 c^2 mu[0]^2 + mu[0]^4) w[0] + a^4 Erfc[mu[0]/(Sqrt[2] c)] (3 c^4 + 6 c^2 mu[0]^2 + mu[0]^4) w[0] + 6 c^4 w[1] + E^(-((mu[1]^2 + mu[2]^2)/(2 c^2))) (-((-1 + a^4) c E^(mu[1]^2/(2 c^2)) Sqrt[2/\[Pi]] mu[2] (5 c^2 + mu[2]^2) w[2]) + E^(mu[2]^2/(2 c^2)) (-((-1 + a^4) c Sqrt[2/\[Pi]] mu[1] (5 c^2 + mu[1]^2) w[1]) + E^(mu[1]^2/(2 c^2)) (2 mu[1]^2 (6 c^2 + mu[1]^2) w[1] + (-1 + a^4) Erfc[mu[1]/(Sqrt[2] c)] (3 c^4 + 6 c^2 mu[1]^2 + mu[1]^4) w[1] + (2 + (-1 + a^4) Erfc[mu[2]/(Sqrt[2] c)]) (3 c^4 + 6 c^2 mu[2]^2 + mu[2]^4) w[2]))))'
# m5 = r'1/2 (10 c^2 mu[0]^3 w[0] + mu[0]^5 w[0] + c^4 (-8 (-1 + a^5) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]] + 15 mu[0]) w[0] - (-1 + a^5) c E^(-(mu[0]^2/(2 c^2))) Sqrt[2/\[Pi]] mu[0]^2 (9 c^2 + mu[0]^2) w[0] + Erf[mu[0]/(Sqrt[2] c)] mu[0] (15 c^4 + 10 c^2 mu[0]^2 + mu[0]^4) w[0] + a^5 Erfc[mu[0]/(Sqrt[2] c)] mu[0] (15 c^4 + 10 c^2 mu[0]^2 + mu[0]^4) w[0] + 30 c^4 mu[1] w[1] + 20 c^2 mu[1]^3 w[1] + 2 mu[1]^5 w[1] - (-1 + a^5) c E^(-(mu[1]^2/(2 c^2))) Sqrt[2/\[Pi]] (c^2 + mu[1]^2) (8 c^2 + mu[1]^2) w[1] + (-1 + a^5) Erfc[mu[1]/(Sqrt[2] c)] mu[1] (15 c^4 + 10 c^2 mu[1]^2 + mu[1]^4) w[1] - (-1 + a^5) c E^(-(mu[2]^2/(2 c^2))) Sqrt[2/\[Pi]] (c^2 + mu[2]^2) (8 c^2 + mu[2]^2) w[2] + (2 + (-1 + a^5) Erfc[mu[2]/(Sqrt[2] c)]) mu[2] (15 c^4 + 10 c^2 mu[2]^2 + mu[2]^4) w[2])'

#These are the formulas for 2 components
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

# Print the processed moment expressions
for i, moment in enumerate(m, start=1):
    print(f"Moment {i}: {moment}")

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


#******Compare error vs. Sample Count******
variances = [0.1, 0.5, 1.0, 2.0]
samplecounts = [100,1000,10000,100000,1000000,10000000]

for c_temp in variances:
    moments_samples_all = []
    for number in samplecounts:
        #Generate random samples from a Gaussian Mixturefrom scikit learn
        gm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full')
        gm.means_ = np.array([[mu0_test], [mu1_test]])
        gm.covariances_ = np.array([[[c_test]], [[c_test]]])
        gm.weights_ = np.array([w0_test, w1_test])
        samples_prior  = gm.sample(1000000)[0]

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

        moments_samples_all.append(moments_samples)

    moments_samples_all = np.array(moments_samples_all)
    moments_samples_all = np.transpose(moments_samples_all)

    #Substitute Values
    values = {a: a_test, c: c_test, mu0: mu0_test, mu1:mu1_test, w0: w0_test, w1: w1_test}
    moments_analytic = []
    for i, mx in enumerate(mp):
        evaluated_expr = mx.subs(values)
        # Numerically evaluate the result
        result = evaluated_expr.evalf()
        moments_analytic.append(result)

    # PLot error vs sample count for every moment
    plt.figure(figsize=(10, 6))
    plt.title(f'Error vs Sample Count for Variance {c_temp}')
    plt.xlabel('Sample Count')
    plt.ylabel('Relative Error (%)')
    for i in range(5):
        errors = []
        for j in range(len(samplecounts)):
            error = abs(moments_samples_all[i][j] - moments_analytic[i]) / abs(moments_samples_all[i][j]) * 100
            errors.append(error)
        plt.plot(samplecounts, errors, label=f'{i+1}. Moment')
    plt.legend()
    plt.grid()
    plt.xscale('log')
    plt.ylim(0,100)
    plt.savefig(f'workbench/error_vs_sample_count_var_{c_temp}_GM.png')