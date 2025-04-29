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

print(">CHANGE EVRYTHING TO GM")

#******Define testing connditions******
np.random.seed(42)  # Set seed for reproducibility

a_test = 0.05
c_test = 1
mu_test = 0.0

#******Sample Based Moment Computation******
#Generate random samples form a gaussian distribution with mean mu and variance c
samples_prior = np.random.normal(mu_test, c_test, 100000)

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
a, c, mu = var('a c mu')

# Gaussian Moment through leaky relu.Expression from mathematica (\[]-Style expressions need to be replaced without brackets)
m = []
m1 = '-(((-1 + a) c E^(-(mu^2/(2 c^2))))/Sqrt[2 Pi]) + 1/2 mu (1 + Erf[mu/(Sqrt[2] c)] + a Erfc[mu/(Sqrt[2] c)])'
m2 = '-(((-1 + a^2) c E^(-(mu^2/(2 c^2))) mu)/Sqrt[2 Pi]) + 1/2 (c^2 + mu^2) (1 + Erf[mu/(Sqrt[2] c)] + a^2 Erfc[mu/(Sqrt[2] c)])'
m3 = '-(((-1 + a^3) c E^(-(mu^2/(2 c^2))) (2 c^2 + mu^2))/Sqrt[2 Pi]) + 1/2 mu (3 c^2 + mu^2) (1 + Erf[mu/(Sqrt[2] c)] + a^3 Erfc[mu/(Sqrt[2] c)])'
m4 = '-(((-1 + a^4) c E^(-(mu^2/(2 c^2))) mu (5 c^2 + mu^2))/Sqrt[2 Pi]) + 1/2 (3 c^4 + 6 c^2 mu^2 + mu^4) (1 + Erf[mu/(Sqrt[2] c)] + a^4 Erfc[mu/(Sqrt[2] c)])'
m5 = '-(((-1 + a^5) c E^(-(mu^2/(2 c^2))) (c^2 + mu^2) (8 c^2 + mu^2))/Sqrt[2 Pi]) + 1/2 mu (15 c^4 + 10 c^2 mu^2 + mu^4) (1 + Erf[mu/(Sqrt[2] c)] + a^5 Erfc[mu/(Sqrt[2] c)])'

m.append(m1)
m.append(m2)
m.append(m3)
m.append(m4)
m.append(m5)

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
values = {a: a_test, c: c_test, mu: mu_test}
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
        #Generate random samples form a gaussian distribution with mean mu and variance c
        samples_prior = np.random.normal(mu_test, c_temp,number)

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
    values = {a: a_test, c: c_temp, mu: mu_test}
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
    plt.savefig(f'workbench/error_vs_sample_count_var_{c_temp}.png')