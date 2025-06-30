#######################################################################################
# Utilities for Gaussian Mixture Network model 
# Author: Leon Winheim, ISAS at KIT Karlsruhe
# Date: 19.05.2025
#######################################################################################
import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import least_squares, LinearConstraint, minimize, Bounds
from math import factorial, comb
import time
from sklearn.mixture import GaussianMixture
from itertools import combinations

tuple_cache = {}
factorial_cache = {}

# Noncentral moments of a Gaussian, flexible
def double_factorial(n):
    """Compute the double factorial n!!"""
    if n <= 0:
        return 1
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result

def gaussian_noncentral_moment(n, mu, c):
    """Compute the n-th noncentral moment of a Gaussian"""
    # Compute stadderd deviation and initialize moment
    sigma = np.sqrt(c)
    moment = 0
    for k in range(n // 2 + 1):
        coeff = comb(n, 2 * k)
        dfact = double_factorial(2 * k - 1)
        term = coeff * dfact * (sigma ** (2 * k)) * (mu ** (n - 2 * k))
        moment += term
    return moment

# Noncentral moments of a Gaussian Mixture for arbitrary many components and order
def gm_noncentral_moment(n, w, mu, c):
    """Compute the n-th noncentral moment of a Gaussian Mixture"""
    moment = 0
    for i in range(len(w)):
        moment += w[i] * gaussian_noncentral_moment(n, mu[i], c[i])
    return moment

# Next generation of functions
def generate_k_tuples(n, m):
    """
    Generate all m-tuples of non-negative integers (k_1,...,k_m) such that sum(k_i) = n.
    Yields one tuple at a time.
    I need this for the multinomial expansion
    """
    if m == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for tail in generate_k_tuples(n - i, m - 1):
                yield (i,) + tail

def generate_k_tuples_fast(n, k):
    positions = range(n + k - 1)
    for dividers in combinations(positions, k - 1):
        counts = []
        last = -1
        for d in dividers:
            counts.append(d - last - 1)
            last = d
        counts.append(n + k - 1 - last - 1)
        yield tuple(counts)

def get_factorial(n):
    """Get the factorial of n, caching the result for future use."""
    if n in factorial_cache:
        #print(f"Using cached factorial for {n}")
        return factorial_cache[n]
    else:
        result = factorial(n)
        factorial_cache[n] = result
        return result

def moments_pre_act_combined_general(z:np.ndarray,w:np.ndarray,order:int=10,moments_z_in = None):
    """ This function computes the first ten moments of the pre activation value for a single neuron with multiple products input and weight.
        The nature of this computation is a multinomial expansion.

        z is an array containing the the input values. Every row is a different GM with [[mu1,mu2...muN],[c1,c2...cN],[w1,w2...wN]] and resembles one neuronal output from before
        w is an array containing tuples of the form (mu_w, c_w) for every entry representing the weight as independent Gaussian
        order is the number of moments to compute (default is 10)
        
        returns an array of the first ten moments of the pre activation value

        Tuples will be cached
    """

    number_moments = order    #If we want to change this, we need to add more moments to the functions above

    # In general, x_array could be a gaussian mixture or deterministic. 
    # If it is deterministic, it will be the value as the first mean, 1 as the first weight and all other values in the GM specification will be zero
    
    assert z.shape[0] == w.shape[0], "z and w must have the same length, but have lengths {} and {}".format(z.shape[0], w.shape[0])  

    # Pre-Compute the Moments of each w (w is always a Gaussian, so this is always needed)
    moments_w = np.zeros((len(w), number_moments+1))
    for i in range(w.shape[0]):
        mu_w, c_w = w[i]
        # Save the moments, augment a 0 moment to avoid indexing issues with the multinomial indexing
        moments_w[i][0] = 1  
        for j in range(1,number_moments+1):
            moments_w[i][j] = gaussian_noncentral_moment(j, mu_w, c_w)

    # Pre-Compute the Moments of each z (z could be Deterministic or GM, but the gm-moment-generation will handle this flexible)
    moments_z = np.zeros((len(z), number_moments+1))
    if moments_z_in is None:
        for i in range(z.shape[0]):
            # mu, c and w are potentially multi-component GMs
            mu_z, c_z, w_z = z[i]
            # Save the moments, augment a 0 moment to avoid indexing issues with the multinomial indexing
            moments_z[i][0] = 1
            for j in range(1,number_moments+1):
                moments_z[i][j] = gm_noncentral_moment(j, w_z, mu_z, c_z)
    else:
        raise ValueError("moments_z_in is not None, this is not implemented yet")

    # Initialize result vector 
    result = np.zeros(number_moments)

    # Computation of the first ten moments
    for i in range(1,number_moments+1):
        # Get multinomial tuples for the desired moment and number of neuron inputs
        if (i, z.shape[0]) in tuple_cache:
            tuples = tuple_cache[(i, z.shape[0])]
        else:
            tuples = np.array(tuple(generate_k_tuples(i, z.shape[0])))
            tuple_cache[(i, z.shape[0])] = tuples
        
        # Iterate through the tuples (I turned it into an array, but still refer to it as value tuples)
        for t in range(len(tuples)):
            # Compute the multinomial coefficient of the current tuple
            multinom_coeff =  get_factorial(i) // np.prod([get_factorial(k) for k in tuples[t,:]])
            # Iterate through the tuple for every x_i and w_i in the pre-activation sum
            expectation_product = 1
            for j in range(z.shape[0]):
                k = tuples[t,j]
                # Do the expectation product
                if moments_z_in is None: 
                    expectation_product *= moments_z[j][k] * moments_w[j][k]
                else:
                    expectation_product *= moments_z[j,k] * moments_w[j][k]
            # Add up all the values for the moments
            result[i-1] += multinom_coeff * expectation_product

    return result

def moments_pre_act_combined_general_peak(z:np.ndarray,w:np.ndarray,dirac:np.ndarray,order:int=10):
    """ This function computes the first ten moments of the pre activation value for a single neuron with multiple products input and weight, and an additional dirac.
        The nature of this computation is a multinomial expansion.

        z is an array containing the the input values. Every row is a different GM with [[mu1,mu2...muN],[c1,c2...cN],[w1,w2...wN]] and resembles one neuronal output from before
        w is an array containing tuples of the form (mu_w, c_w) for every entry representing the weight as independent Gaussian
        order is the number of moments to compute (default is 10)
        
        returns an array of the first ten moments of the pre activation value
    """
    #raise ValueError("Wie baue ich den Dirac da ein?")
    number_moments = order    #If we want to change this, we need to add more moments to the functions above

    # In general, x_array could be a gaussian mixture or deterministic. 
    # If it is deterministic, it will be the value as the first mean, 1 as the first weight and all other values in the GM specification will be zero
    
    assert z.shape[0] == w.shape[0], "z and w must have the same length, but have lengths {} and {}".format(z.shape[0], w.shape[0])  

    # Pre-Compute the moments of each w (w is always a Gaussian, so this is always needed)
    moments_w = np.zeros((len(w), number_moments+1))
    for i in range(w.shape[0]):
        mu_w, c_w = w[i]
        # Save the moments, augment a 0 moment to avoid indexing issues with the multinomial indexing
        moments_w[i][0] = 1  
        for j in range(1,number_moments+1):
            moments_w[i][j] = gaussian_noncentral_moment(j, mu_w, c_w)
    # Pre-Compute the Moments of each z (z could be Deterministic or GM, but the gm-moment-generation will handle this flexible)
    moments_z = np.zeros((len(z), number_moments+1))
    for i in range(z.shape[0]):
        # mu, c and w are potentially multi-component GMs
        mu_z, c_z, w_z = z[i]
        # Save the moments, augment a 0 moment to avoid indexing issues with the multinomial indexing
        moments_z[i][0] = 1
        for j in range(1,number_moments+1):
            moments_z[i][j] = gm_noncentral_moment(j, w_z, mu_z, c_z)
            # ADD THE PEAK DIRAC MOMENT VALUE HERE
            # BUT IT DOESNT EVEN CONTRIBUTE????

    # Initialize result vector 
    result = np.zeros(number_moments)

    # Computation of the first ten moments
    for i in range(1,number_moments+1):
        # Get multinomial tuples for the desired moment and number of neuron inputs
        tuples = np.array(tuple(generate_k_tuples(i, z.shape[0])))

        # Iterate through the tuples (I turned it into an array, but still refer to it as value tuples)
        for t in range(len(tuples)):
            # Compute the multinomial coefficient of the current tuple
            multinom_coeff =  factorial(i) // np.prod([factorial(k) for k in tuples[t,:]])
            # Iterate through the tuple for every x_i and w_i in the pre-activation sum
            expectation_product = 1
            for j in range(z.shape[0]):
                k = tuples[t,j]
                # Do the expectation product
                expectation_product *= moments_z[j][k] * moments_w[j][k]
            # Add up all the values for the moments
            result[i-1] += multinom_coeff * expectation_product
    return result

def moments_post_act(a:float,mu:np.ndarray,c:np.ndarray,w:np.ndarray,order:int=10): 
    """This function computes the post activation moments of a Gaussian mixture with arbitrary many components propagated through leaky relu
    
        a: slope of the leaky relu
        mu: mean vector of the Gaussian Mixture
        c: variance array of the Gaussian Mixture
        w: weights of the Gaussian Mixture
        order: the number of moments to compute (default is 10)

        returns the first ten moments of the post activation distribution
    """
    assert np.max(c) >= 0.01, "Variance must be big enough"

    num_moments = order
    assert len(mu) == len(c) == len(w), "mu, c, and w must have the same length"
    sigma = np.sqrt(c)

    # Initialize result vector 
    result = np.zeros(10, dtype=float)

    # Iterate over components of the GM
    for i in range(len(mu)):
        result[0] += (((1 + erf(mu[i]/(np.sqrt(2)*sigma[i])))*mu[i])/2 +(a*erfc(mu[i]/(np.sqrt(2)*sigma[i]))*mu[i])/2 + sigma[i]/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) - (a*sigma[i])/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)))*w[i]
        result[1] += ((mu[i]*sigma[i])/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) -(a**2*mu[i]*sigma[i])/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) + ((1 + erf(mu[i]/(np.sqrt(2)*sigma[i])))*(mu[i]**2 + sigma[i]**2))/2 + (a**2*erfc(mu[i]/(np.sqrt(2)*sigma[i]))*(mu[i]**2 + sigma[i]**2))/2)*w[i]
        result[2] += ((sigma[i]*(mu[i]**2 + 2*sigma[i]**2))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) - (a**3*sigma[i]*(mu[i]**2 + 2*sigma[i]**2))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) + ((1 + erf(mu[i]/(np.sqrt(2)*sigma[i])))*mu[i]*(mu[i]**2 + 3*sigma[i]**2))/2 + (a**3*erfc(mu[i]/(np.sqrt(2)*sigma[i]))*mu[i]*(mu[i]**2 + 3*sigma[i]**2))/2)*w[i]
        result[3] += ((mu[i]*sigma[i]*(mu[i]**2 + 5*sigma[i]**2))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) - (a**4*mu[i]*sigma[i]*(mu[i]**2 +  5*sigma[i]**2))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) + ((1 + erf(mu[i]/(np.sqrt(2)*sigma[i])))*(mu[i]**4 + 6*mu[i]**2*sigma[i]**2 + 3*sigma[i]**4))/2 + (a**4*erfc(mu[i]/(np.sqrt(2)*sigma[i]))*(mu[i]**4 + 6*mu[i]**2*sigma[i]**2 + 3*sigma[i]**4))/2)*w[i]
        result[4] += ((sigma[i]*(mu[i]**2 + sigma[i]**2)*(mu[i]**2 + 8*sigma[i]**2))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) - (a**5*sigma[i]*(mu[i]**2 + sigma[i]**2)*(mu[i]**2 + 8*sigma[i]**2))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi)) + ((1 + erf(mu[i]/(np.sqrt(2)*sigma[i])))*mu[i]*(mu[i]**4 + 10*mu[i]**2*sigma[i]**2 + 15*sigma[i]**4))/2 + (a**5*erfc(mu[i]/(np.sqrt(2)*sigma[i]))*mu[i]*(mu[i]**4 + 10*mu[i]**2*sigma[i]**2 + 15*sigma[i]**4))/2)*w[i]
        result[5] += ((mu[i] * sigma[i] * (mu[i]**4 + 14 * mu[i]**2 * sigma[i]**2 + 33 * sigma[i]**4)) / (np.exp(mu[i]**2 / (2 * sigma[i]**2)) * np.sqrt(2 * np.pi))- (a**6 * mu[i] * sigma[i] * (mu[i]**4 + 14 * mu[i]**2 * sigma[i]**2 + 33 * sigma[i]**4)) / (np.exp(mu[i]**2 / (2 * sigma[i]**2)) * np.sqrt(2 * np.pi))+ ((1 + erf(mu[i] / (np.sqrt(2) * sigma[i]))) * (mu[i]**6 + 15 * mu[i]**4 * sigma[i]**2 + 45 * mu[i]**2 * sigma[i]**4 + 15 * sigma[i]**6)) / 2+ (a**6 * erfc(mu[i] / (np.sqrt(2) * sigma[i])) * (mu[i]**6 + 15 * mu[i]**4 * sigma[i]**2 + 45 * mu[i]**2 * sigma[i]**4 + 15 * sigma[i]**6)) / 2) * w[i]
        result[6] += ((sigma[i]*(mu[i]**6 + 20*mu[i]**4*sigma[i]**2 + 87*mu[i]**2*sigma[i]**4 + 48*sigma[i]**6))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi))- (a**7*sigma[i]*(mu[i]**6 + 20*mu[i]**4*sigma[i]**2 + 87*mu[i]**2*sigma[i]**4 + 48*sigma[i]**6))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi))+ ((1 + erf(mu[i]/(np.sqrt(2)*sigma[i])))*mu[i]*(mu[i]**6 + 21*mu[i]**4*sigma[i]**2 + 105*mu[i]**2*sigma[i]**4 + 105*sigma[i]**6))/2+ (a**7*erfc(mu[i]/(np.sqrt(2)*sigma[i]))*mu[i]*(mu[i]**6 + 21*mu[i]**4*sigma[i]**2 + 105*mu[i]**2*sigma[i]**4 + 105*sigma[i]**6))/2) * w[i]
        result[7] += ((mu[i] * sigma[i] * (mu[i]**6 + 27 * mu[i]**4 * sigma[i]**2 + 185 * mu[i]**2 * sigma[i]**4 + 279 * sigma[i]**6)) /(np.exp(mu[i]**2 / (2 * sigma[i]**2)) * np.sqrt(2 * np.pi))- (a**8 * mu[i] * sigma[i] * (mu[i]**6 + 27 * mu[i]**4 * sigma[i]**2 + 185 * mu[i]**2 * sigma[i]**4 + 279 * sigma[i]**6)) /(np.exp(mu[i]**2 / (2 * sigma[i]**2)) * np.sqrt(2 * np.pi))+ ((1 + erf(mu[i] / (np.sqrt(2) * sigma[i]))) * (mu[i]**8 + 28 * mu[i]**6 * sigma[i]**2 + 210 * mu[i]**4 * sigma[i]**4 + 420 * mu[i]**2 * sigma[i]**6 + 105 * sigma[i]**8)) / 2+ (a**8 * erfc(mu[i] / (np.sqrt(2) * sigma[i])) * (mu[i]**8 + 28 * mu[i]**6 * sigma[i]**2 + 210 * mu[i]**4 * sigma[i]**4 + 420 * mu[i]**2 * sigma[i]**6 + 105 * sigma[i]**8)) / 2) * w[i]
        result[8] += ((sigma[i]*(mu[i]**8 + 35*mu[i]**6*sigma[i]**2 + 345*mu[i]**4*sigma[i]**4 + 975*mu[i]**2*sigma[i]**6 + 384*sigma[i]**8))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi))- (a**9*sigma[i]*(mu[i]**8 + 35*mu[i]**6*sigma[i]**2 + 345*mu[i]**4*sigma[i]**4 + 975*mu[i]**2*sigma[i]**6 + 384*sigma[i]**8))/(np.exp(mu[i]**2/(2*sigma[i]**2))*np.sqrt(2*np.pi))+ ((1 + erf(mu[i]/(np.sqrt(2)*sigma[i])))*mu[i]*(mu[i]**8 + 36*mu[i]**6*sigma[i]**2 + 378*mu[i]**4*sigma[i]**4 + 1260*mu[i]**2*sigma[i]**6 + 945*sigma[i]**8))/2+ (a**9*erfc(mu[i]/(np.sqrt(2)*sigma[i]))*mu[i]*(mu[i]**8 + 36*mu[i]**6*sigma[i]**2 + 378*mu[i]**4*sigma[i]**4 + 1260*mu[i]**2*sigma[i]**6 + 945*sigma[i]**8))/2) * w[i]
        result[9] += w[i] * ((mu[i] * sigma[i] * (mu[i]**8 + 44 * mu[i]**6 * sigma[i]**2 + 588 * mu[i]**4 * sigma[i]**4 + 2640 * mu[i]**2 * sigma[i]**6 + 2895 * sigma[i]**8)) / (np.exp(mu[i]**2 / (2 * sigma[i]**2)) * np.sqrt(2 * np.pi))+ ((1 + erf(mu[i] / (np.sqrt(2) * sigma[i]))) * (mu[i]**10 + 45 * mu[i]**8 * sigma[i]**2 + 630 * mu[i]**6 * sigma[i]**4 + 3150 * mu[i]**4 * sigma[i]**6 + 4725 * mu[i]**2 * sigma[i]**8 + 945 * sigma[i]**10)) / 2+ (a**10 * (-np.sqrt(2 / np.pi) * mu[i] * sigma[i] * (mu[i]**8 + 44 * mu[i]**6 * sigma[i]**2 + 588 * mu[i]**4 * sigma[i]**4 + 2640 * mu[i]**2 * sigma[i]**6 + 2895 * sigma[i]**8) / np.exp(mu[i]**2 / (2 * sigma[i]**2)) + erfc(mu[i] / (np.sqrt(2) * sigma[i])) * (mu[i]**10 + 45 * mu[i]**8 * sigma[i]**2 + 630 * mu[i]**6 * sigma[i]**4 + 3150 * mu[i]**4 * sigma[i]**6 + 4725 * mu[i]**2 * sigma[i]**8 + 945 * sigma[i]**10)) / 2))
        
    return result[:num_moments]  # Return only the first 8 moments, as the last two are not needed for the network

def match_moments(in_mom,components,solver="trust-constr"):
    """Function to perform the moment matching for given Moments and a desired number of components.
        Solver can be either SLSQP or trust-constr.
    """
    #Fallback
    if components == 2:
        return match_moments_2(in_mom, components)

    # Assemble the parameter vector according to the number of components. Per Component, we have 3 parameters: w, mu, c
    # The complete parameter vector will look like this: [w0,w1...,wN,mu0,mu1,...,muN,c0,c1,...,cN]
    params = np.zeros(components*3, dtype=float)

    #Set the initial guess for the weights as equal and summing up to one
    for i in range(components):
        params[i] = 1/components

    #Set the initial guess for the means as the input moments
    params[components] =-1.8   
    for i in range(components-1):
        params[components+i+1] = float(1.8*i)    # Rest of the means 

    #Set the initial guess for the variances as the input moments
    for i in range(components):
        params[components*2+i] = 0.1    

    # Optimize this with the minimize function to add in the constraint on the sum of the weights

    l_bounds = np.zeros(components*3)                       # Set the bounds for the parameters as 1D array
    l_bounds[components:components*2] = -np.inf             # Lower bounds for the means
    l_bounds[components*2:] = 0.0                           # Lower bounds for the variances

    u_bounds = np.full(components*3, np.inf)                # Set the bounds for the parameters as 1D array
    u_bounds[0:components] = 1.0                            # Upper bounds for the weights
    # Means and variances already set to np.inf
    bounds = Bounds(l_bounds, u_bounds)

    if solver == "trust-constr":
        in_constr = np.zeros((1,components*3))                      #Set a multiplicator for every parameter
        in_constr[0,0:components] = 1.0                             # Only the weights are subject to the constraint
        constraint  = LinearConstraint(in_constr, [1.0], [1.0])     # Constraint on the sum of the weights to 1

        # Call optimizer with bounds and constraints
        start = time.time()	
        result = minimize(residuals_matching_n, 
                            params,
                            args=in_mom,
                            method='trust-constr',
                            bounds=bounds,
                            constraints=[constraint],
                            #options={'disp': True, 'xtol': 1e-12, 'gtol':1e-12, 'maxiter': 100000}
                            #options={'disp': False, 'xtol': 1e-6, 'gtol': 1e-6, 'maxiter': 1000}
                            #options={'disp': False}
                        )
        end = time.time()
        #print("Time for optimization: {}".format(end-start))
    elif solver == "SLSQP":
        constraint_dict = {'type': 'eq',
                        'fun': lambda x: np.sum(x[:components]) - 1.0
                        }
        # Call optimizer with bounds and constraints
        result = minimize(residuals_matching_n, 
                            params,
                            args=in_mom,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=[constraint_dict],
                            #options={'disp': True, 'xtol': 1e-12, 'gtol':1e-12, 'maxiter': 100000}
                            options={'disp': False}
                        )
    else:
        raise ValueError("Solver not supported. Please use either 'SLSQP' or 'trust-constr'")
    
    # print("Result of the optimization:")
    # print(result)
    # print("Final parameters:")
    # print(result.x)
    
    w_res = np.array(result.x[:components])
    mu_res = np.array(result.x[components:components*2])
    c_res = np.array(result.x[components*2:])

    assert np.isclose(np.sum(w_res), 1.0, atol=1e-6), f"Weights do not sum to 1.0 (sum={np.sum(w_res)})"

    return mu_res, c_res, w_res

def match_moments_2(in_mom, components):
    """Intermediate solution, only for two components"""
    start = time.time()
    if components != 2:
        raise ValueError("This function only works for two components")
    
    #params = [0.5,0.0,5.0,1.0,1.0]  # Initial guess for [w0, mu0, mu1, c0, c1]
    params = [0.5,0.0,1.0,0.1,0.1]  # Initial guess for [w0, mu0, mu1, c0, c1]

    # Call optimizer with bounds
    start = time.time()
    result = least_squares(residuals_matching_2, params, args=in_mom, bounds=([0, -np.inf, -np.inf, 0, 0], [1.0, np.inf, np.inf, np.inf, np.inf]))
    end = time.time()
    #print("Residuals:", result.fun)

    w_res = [result.x[0], 1 - result.x[0]]
    mu_res = [result.x[1], result.x[2]]
    c_res = [result.x[3], result.x[4]]
    end = time.time()

    assert np.isclose(np.sum(w_res), 1.0, atol=1e-6), f"Weights do not sum to 1.0 (sum={np.sum(w_res)})"
    assert w_res[0] >= 0 and w_res[1] >= 0, "Weights must be non-negative"

    return np.array(mu_res), np.array(c_res), np.array(w_res)

def match_moments_peak(in_mom, in_mass, components, solver="trust-constr"):
    """ 
    Function to perform the moment matching to a Gauzss Mixture in Combination with a fixed peak at 0.
    The peak represents the probability mass that was below zero before the ReLU transformation.
    We optimize the parameters of the Gaussian Mixture to match the moments of the input distribution, 
    keeping the dirac in mind.

    """

    # Assemble the parameter vector according to the number of components. Per Component, we have 3 parameters: w, mu, c
    # The complete parameter vector will look like this: [w0,w1...,wN,mu0,mu1,...,muN,c0,c1,...,cN]
    params = np.zeros(components*3, dtype=float)

    #Set the initial guess for the weights as equal and summing up to one-dirac mass
    for i in range(components):
        params[i] = (1-in_mass)/components

    #Set the initial guess for the means as the input moments
    params[components] = 0.1    
    for i in range(components-1):
        params[components+i+1] = float(1*i)    # Rest of the means 

    #Set the initial guess for the variances as the input moments
    for i in range(components):
        params[components*2+i] = 0.1    
    # Optimize this with the minimize function to add in the constraint on the sum of the weights

    l_bounds = np.zeros(components*3)                       # Set the bounds for the parameters as 1D array
    l_bounds[components:components*2] = -np.inf             # Lower bounds for the means
    l_bounds[components*2:] = 0.0                           # Lower bounds for the variances

    u_bounds = np.full(components*3, np.inf)                # Set the bounds for the parameters as 1D array
    u_bounds[0:components] = 1.0                            # Upper bounds for the weights
    # Means and variances already set to np.inf
    bounds = Bounds(l_bounds, u_bounds)

    if solver == "trust-constr":
        in_constr = np.zeros((1,components*3))                      #Set a multiplicator for every parameter
        in_constr[0,0:components] = 1.0                             # Only the weights are subject to the constraint
        constraint  = LinearConstraint(in_constr, [1.0-in_mass], [1.0-in_mass])     # Constraint on the sum of the weights to 1-in_mass

        # Call optimizer with bounds and constraints
        start = time.time()	
        result = minimize(residuals_matching_peak, 
                            params,
                            args=[*in_mom,0,in_mass],
                            method='trust-constr',
                            bounds=bounds,
                            constraints=[constraint],
                            #options={'disp': True, 'xtol': 1e-12, 'gtol':1e-12, 'maxiter': 100000}
                            #options={'disp': False, 'xtol': 1e-6, 'gtol': 1e-6, 'maxiter': 1000}
                            #options={'disp': False}
                        )
        end = time.time()
        #print("Time for optimization: {}".format(end-start))
    elif solver == "SLSQP":
        constraint_dict = {'type': 'eq',
                        'fun': lambda x: np.sum(x[:components]) - 1.0
                        }
        # Call optimizer with bounds and constraints
        result = minimize(residuals_matching_peak, 
                            params,
                            args=in_mom,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=[constraint_dict],
                            #options={'disp': True, 'xtol': 1e-12, 'gtol':1e-12, 'maxiter': 100000}
                            options={'disp': False}
                        )
    else:
        raise ValueError("Solver not supported. Please use either 'SLSQP' or 'trust-constr'")
    
    # print("Result of the optimization:")
    # print(result)
    # print("Final parameters:")
    # print(result.x)
    
    w_res = np.array(result.x[:components])
    mu_res = np.array(result.x[components:components*2])
    c_res = np.array(result.x[components*2:])

    #assert np.isclose(np.sum(w_res), 1.0, atol=1e-6), f"Weights do not sum to 1.0 (sum={np.sum(w_res)})"

    return mu_res, c_res, w_res

def residuals_matching_n(params, *args):
    """
        Compute the residual value for the optimization process.
        Returns the array of indivudal residuals for each moment.
        This is for arbitrary component count
    """
    # Unpack the arguments
    t = []
    for temp in args:
        t.append(temp)
    t = np.array(t).squeeze()	

    num_moments = len(t)  # Number of moments to match

    # Infer how many components we have
    components = int(len(params)/3)

    # Extract the parameters from the input vector
    w = params[:components]
    mu = params[components:components*2]
    c = params[components*2:components*3]

    # Compute the moments of the Gaussian Mixture
    gm_moments = np.zeros(num_moments, dtype=float)
    for i in range(num_moments):
        gm_moments[i] = gm_noncentral_moment(i+1, w, mu, c)

    # Compute the weighted residuals
    residuals = np.zeros(num_moments, dtype=float)
    for i in range(num_moments):
        residuals[i] = abs(gm_moments[i] - t[i])/t[i]
    
    # COmpute the summed squared residuals
    residuals = np.sum(residuals**2)

    # This should return an array
    return residuals

def residuals_matching_2(params, *args):
    """
        Compute the residual value for the optimization process.
        Returns the array of indivudal residuals for each moment.
        This is for two component only
    """

    # Unpack the arguments
    t = []
    for temp in args:
        t.append(temp)

    num_moments = len(t)  # Number of moments to match

    # Infer how many components we have
    components = int(len(params)/3)
    # Extract the parameters from the input vector
    w = np.array([params[:1].squeeze(),1-params[:1].squeeze()])
    mu = params[1:3]
    c = params[3:5]

    # Compute the moments of the Gaussian Mixture
    gm_moments = np.zeros(num_moments, dtype=float)
    for i in range(num_moments):
        gm_moments[i] = gm_noncentral_moment(i+1, w, mu, c)

    # Compute the weighted residuals
    residuals = np.zeros(num_moments, dtype=float)
    for i in range(num_moments):
        residuals[i] = abs(gm_moments[i] - t[i])/t[i]

    # This should return an array
    return residuals

def residuals_matching_peak(params, *args):
    """
    Compute the residual value for the optimization process.
    Returns the array of indivudal residuals for each moment.
    This is for arbitrary component count, and with a additional dirac component
    """
    # Unpack the arguments
    t = []
    for temp in args:
        t.append(temp)
    t = np.array(t).squeeze()	

    weight_dirac = t[-1]  # The last arg is the weight of the dirac component
    place_dirac = t[-2]  # The second last arg is the place of the dirac component
    num_moments = len(t) - 2  # Number of moments to match, excluding the dirac weight

    # Infer how many components we have
    components = int(len(params)/3)

    # Extract the parameters from the input vector
    w = params[:components]
    mu = params[components:components*2]
    c = params[components*2:components*3]

    # Compute the moments of the Gaussian Mixture /Dirac hybrid
    gm_moments = np.zeros(num_moments, dtype=float)
    for i in range(num_moments):
        gm_moments[i] = gm_noncentral_moment(i+1, w, mu, c)+weight_dirac*place_dirac**(i+1)

    # Compute the weighted residuals
    residuals = np.zeros(num_moments, dtype=float)
    for i in range(num_moments):
        residuals[i] = abs(gm_moments[i] - t[i])/t[i]
    
    # COmpute the summed squared residuals
    residuals = np.sum(residuals**2)

    # This should return an array
    return residuals

def match_samples_em(in_samples, components):
    """
    Function to perform the moment matching to a Gaussian Mixture using the EM algorithm.
    The input is a set of samples, and the output is the parameters of the Gaussian Mixture.
    """
    # Initialize the Gaussian Mixture model
    gmm = GaussianMixture(n_components=components, covariance_type='full')

    # Fit the model to the samples
    gmm.fit(in_samples.reshape(-1, 1))

    # Extract the parameters
    weights = gmm.weights_
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()

    return means, covariances, weights

def sample_gm(mu, c, w, num_samples=1000):
    """
    Sample from a Gaussian Mixture model with given means, variances, and weights.
    
    mu: Means of the Gaussian components.
    c: Variances of the Gaussian components.
    w: Weights of the Gaussian components.
    num_samples: Number of samples to generate.
    
    Returns:
        samples: Array of sampled values from the Gaussian Mixture model.
    """
    # Check if the input arrays are valid
    if len(mu) != len(c) or len(mu) != len(w):
        raise ValueError("Means, variances, and weights must have the same length.")

    # Sample from the Gaussian Mixture model
    samples = np.zeros(num_samples)
    for i in range(num_samples):
        component = np.random.choice(len(mu), p=w)
        samples[i] = np.random.normal(mu[component], np.sqrt(c[component]))

    return samples
#########################################################################################   
# Define the network class to build actual architecture

class GaussianMixtureNetwork():
    """
    Bayesian Neural Network based on Gaussian weights and Gaussian Mixture intermediate approximations, based on higher order moment matching.
    """
    def __init__(self,layers:list,activations:list,components_pre:int,components_post:int,moments_pre:int=10,moments_post:int=10,a_relu:float=0.01, peak:bool=False):
        """
        Initialize the Gaussian Mixture Network with the given parameters.

        layers: List of integers representing the number of neurons in each layer. First is Input feautres, last is output features
        activations: List of activation functions for each layer. Contains one entry less than layers
        components_pre: Number of components for the pre-activation layer.
        components_post: Number of components for the post-activation layer.
        moments_pre: Number of moments to use in matching for the pre-activation layer.
        moments_post: Number of moments to use in matching for the post-activation layer.
        a_relu: Slope of the ReLU activation function, used for the analytic moments.
        peak: Boolean indicating whether to use a peak in the network, which will add a Dirac component at 0 for every neuron matching.
        """
        # Add values to instance
        self.layers = layers                   
        self.activations = activations.copy()
        self.components_pre = components_pre
        self.components_post = components_post
        self.moments_pre = moments_pre
        self.moments_post = moments_post
        self.a_relu = a_relu
        self.verif_samples = 1000000     # Number of samples to use for verificatioon forward poass based on propagated samples
        self.verif_samples_em = 100000   # Number of samples to use for verificatioon forward poass based on propagated samples for EM based moment matching
        self.peak = peak

        # Intialize the weights and biases
        self.init_parameters()      # For Analytic moment matching
        self.init_parameters_em()   # For EM based moment matching

        # Reset activation functions
        self.set_act_func()

        # Print out the network structure
        self.print_network()

    def init_parameters(self):
        """Initialize the structure and the prior parameters of the network"""
        # Initialize weight and bias values
        self.weight_means = []
        self.weight_variances = []
        # Iterate over the layers
        for i in range(len(self.layers)-1):
            # Initialize the weights for each layer. Add 1 for the bias
            weight_mean = np.random.rand(self.layers[i]+1, self.layers[i+1])
            self.weight_means.append(weight_mean)
            weight_variance = np.ones((self.layers[i]+1, self.layers[i+1]))
            self.weight_variances.append(weight_variance)

        # Initialize the containers for the GM parameters in pre-activation (and potentially Dirac Weight)
        self.means_gm_pre = []
        self.variances_gm_pre = []
        self.weights_gm_pre = []
        # Iterate over the layers
        for i in range(len(self.layers)-1):
            # Append empty lists for each layer
            self.means_gm_pre.append(np.zeros((self.layers[i+1], self.components_pre)))
            self.variances_gm_pre.append(np.zeros((self.layers[i+1], self.components_pre)))
            self.weights_gm_pre.append(np.zeros((self.layers[i+1], self.components_pre)))

        # Initialize the containers for the GM parameters in post-activation (and potentially Dirac Weight)
        self.means_gm_post = []
        self.variances_gm_post = []
        self.weights_gm_post = []
        self.dirac_weight_post = []
        # Iterate over the layers
        for i in range(len(self.layers)-1):
            # Append empty lists for each layer
            self.means_gm_post.append(np.zeros((self.layers[i+1], self.components_post)))
            self.variances_gm_post.append(np.zeros((self.layers[i+1], self.components_post)))
            self.weights_gm_post.append(np.zeros((self.layers[i+1], self.components_post)))
            self.dirac_weight_post.append(np.zeros((self.layers[i+1])))

        # Generate pre-activation moment container for analytic
        self.pre_activation_moments_analytic = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_pre))

            self.pre_activation_moments_analytic.append(samples)
        
        # Generate post-activation moment container for analytic
        self.post_activation_moments_analytic = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_post))

            self.post_activation_moments_analytic.append(samples)

        # Generate pre-activation moment container for analytic
        self.pre_activation_moments_fitted = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_pre))

            self.pre_activation_moments_fitted.append(samples)
        
        # Generate post-activation moment container for analytic
        self.post_activation_moments_fitted = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_post))

            self.post_activation_moments_fitted.append(samples)

    def init_parameters_em(self):
        """Initialize the structure and the prior parameters of the network FOR EM BASED COMPUTATION"""
        # Initialize weight and bias values
        self.weight_means_em = []
        self.weight_variances_em = []
        # # Iterate over the layers
        # for i in range(len(self.layers)-1):
        #     # Initialize the weights for each layer. Add 1 for the bias
        #     weight_mean = np.random.rand(self.layers[i]+1, self.layers[i+1])
        #     self.weight_means_em.append(weight_mean)
        #     weight_variance = np.ones((self.layers[i]+1, self.layers[i+1]))
        #     self.weight_variances_em.append(weight_variance)
        self.weight_means_em = self.weight_means.copy()  # Use the same means as for the analytic matching
        self.weight_variances_em = self.weight_variances.copy()  # Use the same variances as for the analytic matching

        # Initialize the containers for the GM parameters in pre-activation (and potentially Dirac Weight)
        self.means_gm_pre_em = []
        self.variances_gm_pre_em = []
        self.weights_gm_pre_em = []
        # Iterate over the layers
        for i in range(len(self.layers)-1):
            # Append empty lists for each layer
            self.means_gm_pre_em.append(np.zeros((self.layers[i+1], self.components_pre)))
            self.variances_gm_pre_em.append(np.zeros((self.layers[i+1], self.components_pre)))
            self.weights_gm_pre_em.append(np.zeros((self.layers[i+1], self.components_pre)))

        # Initialize the containers for the GM parameters in post-activation (and potentially Dirac Weight)
        self.means_gm_post_em = []
        self.variances_gm_post_em = []
        self.weights_gm_post_em = []
        self.dirac_weight_post_em = []
        # Iterate over the layers
        for i in range(len(self.layers)-1):
            # Append empty lists for each layer
            self.means_gm_post_em.append(np.zeros((self.layers[i+1], self.components_post)))
            self.variances_gm_post_em.append(np.zeros((self.layers[i+1], self.components_post)))
            self.weights_gm_post_em.append(np.zeros((self.layers[i+1], self.components_post)))
            self.dirac_weight_post_em.append(np.zeros((self.layers[i+1])))

        # Generate pre-activation moment container for analytic
        self.pre_activation_moments_em = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_pre))

            self.pre_activation_moments_em.append(samples)
        
        # Generate post-activation moment container for analytic
        self.post_activation_moments_em = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_post))

            self.post_activation_moments_em.append(samples)

    def sample_weights(self):
        """Generate samples of every weight and bias in the network, mainly needed for verification"""
        # Generate weight samples
        self.weight_samples = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            # Shape: (verif_samples, in_features+1, out_features)
            samples = np.zeros((self.verif_samples, self.layers[i]+1, self.layers[i+1]))
            for row in range(self.layers[i]+1):
                for col in range(self.layers[i+1]):
                    samples[:, row, col] = np.random.normal(
                    self.weight_means[i][row, col],
                    self.weight_variances[i][row, col],
                    self.verif_samples
                    )
            self.weight_samples.append(samples)
        
        # Generate post-activation sample container
        self.post_activation_samples = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            # Shape: (verif_samples, number of neurons in the layer+1)
            samples = np.zeros((self.verif_samples, self.layers[i+1]+1))
            #Turn the Bias samples onto ones
            samples[:, -1] = 1

            self.post_activation_samples.append(samples)

        # Generate pre-activation sample container^
        self.pre_activation_samples = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            # Shape: (verif_samples, number of neurons in the layer)
            samples = np.zeros((self.verif_samples, self.layers[i+1]))
            # We don't save an extra bias here, this will only come in the post activation state

            self.pre_activation_samples.append(samples)

        # Generate pre-activation moment container
        self.pre_activation_moments_samples = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_pre))

            self.pre_activation_moments_samples.append(samples)
        
        # Generate post-activation moment container
        self.post_activation_moments_samples = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_post))

            self.post_activation_moments_samples.append(samples)

    def sample_weights_em(self):
        """Generate samples of every weight and bias in the network, mainly needed for verification"""
        # Generate weight samples
        self.weight_samples_em = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            # Shape: (verif_samples, in_features+1, out_features)
            samples = np.zeros((self.verif_samples_em, self.layers[i]+1, self.layers[i+1]))
            for row in range(self.layers[i]+1):
                for col in range(self.layers[i+1]):
                    samples[:, row, col] = np.random.normal(
                    self.weight_means_em[i][row, col],
                    self.weight_variances_em[i][row, col],
                    self.verif_samples_em
                    )
            self.weight_samples_em.append(samples)
        
        # Generate post-activation sample container
        self.post_activation_samples_em = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            # Shape: (verif_samples, number of neurons in the layer+1)
            samples = np.zeros((self.verif_samples_em, self.layers[i+1]+1))
            #Turn the Bias samples onto ones
            samples[:, -1] = 1

            self.post_activation_samples_em.append(samples)

        # Generate pre-activation sample container^
        self.pre_activation_samples_em = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            # Shape: (verif_samples, number of neurons in the layer)
            samples = np.zeros((self.verif_samples_em, self.layers[i+1]))
            # We don't save an extra bias here, this will only come in the post activation state

            self.pre_activation_samples_em.append(samples)

        # Generate pre-activation moment container
        self.pre_activation_moments_em = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_pre))

            self.pre_activation_moments_em.append(samples)
        
        # Generate post-activation moment container
        self.post_activation_moments_em = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],self.moments_post))

            self.post_activation_moments_em.append(samples)

    def set_act_func(self):
        """Set the activation functions for the network from string to handle"""
        # Initialize the activation functions
        self.activation_functions = []
        for i in range(len(self.layers)-1):
            if self.activations[i] == 'relu':
                self.activation_functions.append(self.relu)
            elif self.activations[i] == 'linear':
                self.activation_functions.append(self.linear)
            elif self.activations[i] == 'sigmoid':
                self.activation_functions.append(self.sigmoid)
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]} but should be {'relu'}")
            
    def relu(self, x, return_name=False):
        """General ReLU activation function with leaky parameter"""
        if return_name:
            return "relu"
        a = self.a_relu  
        return np.where(x > 0, x, a * x)
    
    def linear(self, x, return_name=False):  
        """Linear activation function wioth slope 1"""
        if return_name:
            return "linear"
        return x

    def sigmoid(self, x, return_name=False):
        """Sigmoid activation function"""
        if return_name:
            return "sigmoid"
        return 1 / (1 + np.exp(-x))

    def print_network(self):
        """
        Print out the network structure with activation functions.
        """
        print("Gaussian Mixture Network Structure:")
        for i in range(len(self.layers) - 1):
            act = self.activations[i] if i < len(self.activations) else "none"
            print(f" Layer {i}: {self.layers[i]} -> {self.layers[i+1]} neurons | Activation: {act}")
        print(f" Pre-activation GM components: {self.components_pre}, matched with {self.moments_pre} moments")
        print(f" Post-activation GM components: {self.components_post}, matched with {self.moments_post} moments")
        print(f" Leaky ReLU slope (a): {self.a_relu}")
    
    def compute_absorbed_mass(self, w, mu, c):
        """
        Compute the absorbed mass (probability mass below zero) for a Gaussian Mixture.
        Args:
            w: weights of the Gaussian mixture (array-like)
            mu: means of the Gaussian mixture (array-like)
            c: variances of the Gaussian mixture (array-like)
        Returns:
            absorbed_mass: total probability mass below zero
        """
        sigma = np.sqrt(c)
        sigma = np.where(sigma == 0, 1e-12, sigma)
        absorbed_mass = np.sum(w * 0.5 * (1 + erf(-mu / (np.sqrt(2) * sigma))))
        return absorbed_mass

    def compare_sample_moments_forward(self,x):
        """Makes a forward pass in both implemented methods and compares the moments"""
        print("---VERIFICATION---")
        print()
        start = time.time()
        result = self.forward_samples_ref(x)
        stop = time.time()
        print(f"Time for forward_samples: {stop-start:.2f} s")

        start = time.time()
        result = self.forward_moments_ref(x)
        stop = time.time()
        print(f"Time for momentmatch: {stop-start:.2f} s")

        start = time.time()
        result = self.forward_em(x)
        stop = time.time()
        print(f"Time for EM : {stop-start:.2f} s")

        # for i in range(len(self.pre_activation_moments_analytic)):
        #     rel_pre = np.round(100*(self.pre_activation_moments_analytic[i]-self.pre_activation_moments_samples[i])/self.pre_activation_moments_samples[i],2)

        #     rel_post = np.round(100*(self.post_activation_moments_analytic[i]-self.post_activation_moments_samples[i])/self.post_activation_moments_samples[i],2)

        #     print()
        #     print(f"******Layer {i}******")
        #     print(f"Max. Pre activation rel. error: {np.max(abs(rel_pre))} %; Max. First Moment rel. error: {np.max(abs(rel_pre[:,0]))} %")
        #     print(f"Max. Post activation rel. error: {np.max(abs(rel_post))} %; Max. First Moment rel. error: {np.max(abs(rel_post[:,0]))} %")

        for i in range(len(self.pre_activation_moments_analytic)):
            rel_pre_analytic = np.round(100 * (self.pre_activation_moments_analytic[i] - self.pre_activation_moments_samples[i]) / (self.pre_activation_moments_samples[i] + 1e-12), 2)
            rel_post_analytic = np.round(100 * (self.post_activation_moments_analytic[i] - self.post_activation_moments_samples[i]) / (self.post_activation_moments_samples[i] + 1e-12), 2)
            rel_pre_em = np.round(100 * (self.pre_activation_moments_em[i] - self.pre_activation_moments_samples[i]) / (self.pre_activation_moments_samples[i] + 1e-12), 2)
            rel_post_em = np.round(100 * (self.post_activation_moments_em[i] - self.post_activation_moments_samples[i]) / (self.post_activation_moments_samples[i] + 1e-12), 2)

            print()
            print(f"******Layer {i}******")
            print(f"Analytic vs Samples:")
            print(f"  Max. Pre activation rel. error: {np.max(abs(rel_pre_analytic))} %; Max. First Moment rel. error: {np.max(abs(rel_pre_analytic[:,0]))} %")
            print(f"  Max. Post activation rel. error: {np.max(abs(rel_post_analytic))} %; Max. First Moment rel. error: {np.max(abs(rel_post_analytic[:,0]))} %")
            print(f"EM vs Samples:")
            print(f"  Max. Pre activation rel. error: {np.max(abs(rel_pre_em))} %; Max. First Moment rel. error: {np.max(abs(rel_pre_em[:,0]))} %")
            print(f"  Max. Post activation rel. error: {np.max(abs(rel_post_em))} %; Max. First Moment rel. error: {np.max(abs(rel_post_em[:,0]))} %")

    def compare_sample_moments_forward_special(self,x):
        """Makes a forward pass in both implemented methods and compares the moments"""
        print("---VERIFICATION---")
        print()
        start = time.time()
        result = self.forward_samples_ref(x)
        stop = time.time()
        print(f"Time for forward_samples: {stop-start:.2f} s")

        start = time.time()
        result = self.forward_moments_ref(x)
        stop = time.time()
        print(f"Time for momentmatch: {stop-start:.2f} s")

        pass

    def forward_em(self,x):
        """
        Forward pass through the network based on EM.
        x: Input data, deterministic
        """
        # Generate sample representation of every weight and prepare the intermediate sample values
        self.sample_weights_em()

        if isinstance(x,float) or isinstance(x,int):    
            x = np.array([[x]])

        assert len(x) == self.layers[0], f"Input data must have {self.layers[0]} features, but got {len(x)}"

        # Augment Bias 
        x = np.concatenate((x, np.ones((1, 1))), axis=0)

        # Indexing a GM Parameter value is done by [layer][neuron,component]

        ######
        # Handle the first layer seperately as it uses deterministic input
        ######
        # Note: I assume that the first ectivation is ReLU
        # Iterate over the neurons in the first layer
        for i in range(self.layers[1]):
            # Compute pre avtivation 
            # pre_act_samples shape: (verif_samples,)
            pre_act_samples = np.dot(x.squeeze(), self.weight_samples_em[0][:, :, i].T)

            # REAPPROXIMATE THE SAMPLE SET WITH GM
            means, variances, weights = match_samples_em(pre_act_samples, self.components_pre)
            self.means_gm_pre_em[0][i,:] = means
            self.variances_gm_pre_em[0][i,:] = variances
            self.weights_gm_pre_em[0][i,:] = weights
            # RESAMPLE FROM THE APPROXIMATED GM
            pre_act_samples_re = sample_gm(means, variances, weights, self.verif_samples_em)

            # Store the samples in the pre activation sample container
            self.pre_activation_samples_em[0][:, i] = pre_act_samples_re

            # Compute the empirical moments of the sample set
            moments = np.zeros(self.moments_pre)
            for order in range(1, self.moments_pre+1):
                moments[order-1] = np.mean(pre_act_samples_re**order)

            self.pre_activation_moments_em[0][i, :] = moments

            # Propagate the samples through Activation function
            post_act_samples = self.activation_functions[0](pre_act_samples_re)

            # REAPPROXIMATE THE SAMPLE SET WITH GM
            means, variances, weights = match_samples_em(post_act_samples, self.components_post)
            self.means_gm_post_em[0][i,:] = means
            self.variances_gm_post_em[0][i,:] = variances
            self.weights_gm_post_em[0][i,:] = weights
            # RESAMPLE FROM THE APPROXIMATED GM
            post_act_samples_re = sample_gm(means, variances, weights, self.verif_samples_em)

            # Compute the empirical moments of the sample set
            moments = np.zeros(self.moments_post)
            for order in range(1, self.moments_post+1):
                moments[order-1] = np.mean(post_act_samples_re**order)

            self.post_activation_moments_em[0][i, :] = moments

            # Store the samples in the post activation sample container
            self.post_activation_samples_em[0][:, i] = post_act_samples_re


        ######
        # Iterate over the rest of the layers
        # I need to handle the bias term seperately as it is (Gauss x Deterministic)
        ######
        for l in range(1, len(self.layers)-1):
            # Compute pre avtivation 
            # pre_act_samples shape: (verif_samples,neurons)
            # Append the post activation samples with from before with bias samples
            pre_act_samples = np.einsum('bi,bij->bj', self.post_activation_samples_em[l-1], self.weight_samples_em[l])

            for i in range(self.layers[l+1]):
                # FOR EVERY NEURON; APPROXIMATE THE SAMPLE SET WITH GM
                means, variances, weights = match_samples_em(pre_act_samples[:,i], self.components_pre)
                self.means_gm_pre_em[l][i,:] = means
                self.variances_gm_pre_em[l][i,:] = variances
                self.weights_gm_pre_em[l][i,:] = weights
                # RESAMPLE FROM THE APPROXIMATED GM
                pre_act_samples_re = sample_gm(means, variances, weights, self.verif_samples_em)

                # Store the samples in the pre activation sample container
                self.pre_activation_samples_em[l][:,i] = pre_act_samples_re

                # Compute the empirical moments of the sample set
                moments = np.zeros(self.moments_pre)
                for order in range(1, self.moments_pre+1):
                    moments[order-1] = np.mean(pre_act_samples_re**order)

                self.pre_activation_moments_em[l][i, :] = moments

                # Propagate Samples through Activation function
                post_act_samples = self.activation_functions[l](pre_act_samples_re)
                # REAPPROXIMATE THE SAMPLE SET WITH GM
                means, variances, weights = match_samples_em(post_act_samples, self.components_post)
                self.means_gm_post_em[l][i,:] = means
                self.variances_gm_post_em[l][i,:] = variances
                self.weights_gm_post_em[l][i,:] = weights
                # RESAMPLE FROM THE APPROXIMATED GM
                post_act_samples_re = sample_gm(means, variances, weights, self.verif_samples_em)

                # Store the samples in the post activation sample container
                self.post_activation_samples_em[l][:,i] = post_act_samples_re

                # Compute the empirical moments of the sample set
                moments = np.zeros(self.moments_post)
                for order in range(1, self.moments_post+1):
                    moments[order-1] = np.mean(post_act_samples_re**order)

                self.post_activation_moments_em[l][i, :] = moments

        ######
        return self.post_activation_samples_em[-1]
    
    def forward_moments_ref(self,x):
        """
        Forward pass through the network based on the method of moments.
        Refactored Version
        x: Input data, deterministic
        """
        # Handle scalar input in case it is given as a float or int
        if isinstance(x,float) or isinstance(x,int):    
            x = np.array([[x]])

        assert len(x) == self.layers[0], f"Input data must have {self.layers[0]} features, but got {len(x)}"

        # Indexing a GM Parameter value is done by [layer][neuron,component]

        ######
        # Loop over the layers
        ######
        for l in range(len(self.layers)-1):
            # Handle the bias Augmentation and structuring
            if l == 0:
                # First Layer is determinstic Special Case
                x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
                z_complete = np.stack((x, np.zeros((x.shape[0], x.shape[1])), np.ones((x.shape[0], x.shape[1]))), axis=1) # Emulate a GM behsavior with mean=det, var=0 and weight=1
            else:
                # Attention: The bias is represented as a 1-component GM. The other entries have probably more
                z_complete = np.stack((self.means_gm_post[l-1][:, :], self.variances_gm_post[l-1][:, :], self.weights_gm_post[l-1][:, :]),axis=1)
                z_complete = np.concatenate((z_complete, np.zeros((1, 3, self.components_post))), axis=0)
                z_complete[-1, 0, 0] = 1 # Bias as fictive GM component with mean 1
                z_complete[-1, 1, 0] = 0 # and variance 0
                z_complete[-1, 2, 0] = 1 # and weight 1
            
            ######
            # Loop over the neurons in the layer
            ######
            for i in range(self.layers[l+1]):
                print(f"Layer {l}, Neuron {i} of {self.layers[l+1]}")
                # PRE ACTIVATION
                # Assemble the weight array for the current neuron (Mean and Variance)
                w_array = np.stack((self.weight_means[l][:,i], self.weight_variances[l][:,i]), axis=1)
                # Compute Pre-Activation moments
                print("Compute Pre-Activation Moments...")
                moments_pre = moments_pre_act_combined_general(z_complete,w_array,order=self.moments_pre)
                print("Done.")
                self.pre_activation_moments_analytic[l][i,:] = moments_pre
                # Match the Pre-Activation Moments to a GM
                means, variances, weights = match_moments(moments_pre, components = self.components_pre)
                self.means_gm_pre[l][i,:] = means
                self.variances_gm_pre[l][i,:] = variances
                self.weights_gm_pre[l][i,:] = weights
                # Compute the fitted moments from the GM (as a backup)
                for tt in range(self.moments_pre):
                    self.pre_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)
                
                # POST ACTIVATION
                if self.activations[l] == 'relu':
                    # Compute moments of the post activation
                    moments_post =  moments_post_act(self.a_relu, self.means_gm_pre[l][i,:], self.variances_gm_pre[l][i,:], self.weights_gm_pre[l][i,:],order=self.moments_post)
                    self.post_activation_moments_analytic[l][i,:] = moments_post

                    # Matching with Peak of absorbed mass
                    if self.peak:                        
                        # If we set peak, we need to match the moments with a Dirac component at 0
                        absorbed_mass = self.compute_absorbed_mass(self.weights_gm_pre[l][i,:], self.means_gm_pre[l][i,:], self.variances_gm_pre[l][i,:])
                        means, variances, weights = match_moments_peak(moments_post, absorbed_mass, components = self.components_post)
                        self.means_gm_post[l][i,:] = means
                        self.variances_gm_post[l][i,:] = variances
                        self.weights_gm_post[l][i,:] = weights
                        self.dirac_weight_post[l][i] = absorbed_mass  # Set the Dirac weight to the absorbed mass      
                    
                        # Compute the fitted moments from the GM (as a backup)
                        for tt in range(self.moments_post):
                            self.post_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)
                    
                    # Conventional matching, no peak
                    else:                        
                        # Match the GM parameters to the moments
                        means, variances, weights = match_moments(moments_post, components = self.components_post)
                        self.means_gm_post[l][i,:] = means
                        self.variances_gm_post[l][i,:] = variances
                        self.weights_gm_post[l][i,:] = weights

                        # Compute the fitted moments from the GM (as a backup)
                        for tt in range(self.moments_post):
                            self.post_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)
                    

                elif self.activations[l] == 'linear':
                    # Just take the pre moments as the transformation is linear with slope 1
                    moments_post = moments_pre
                    self.post_activation_moments_analytic[l][i,:] = moments_post   

                    # Match the GM parameters to the moments
                    means, variances, weights = match_moments(moments_post, components = self.components_post)
                    self.means_gm_post[l][i,:] = means
                    self.variances_gm_post[l][i,:] = variances
                    self.weights_gm_post[l][i,:] = weights

                    # Compute the fitted moments from the GM (as a backup)
                    for tt in range(self.moments_post):
                        self.post_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)

                else:
                    raise ValueError(f"Unsupported activation function: {self.activations[l]} but should be {'relu'} or {'linear'}")

        ######
        return self.means_gm_post[-1], self.variances_gm_post[-1], self.weights_gm_post[-1]

    def forward_samples_ref(self,x):
        """
        Forward pass through the network based on samples.
        Refactored Version.
        x: Input data, deterministic
        """
        # Generate sample representationj of every weight and prepare the intermediate sample values
        self.sample_weights()

        if isinstance(x,float) or isinstance(x,int):    
            x = np.array([[x]])

        assert len(x) == self.layers[0], f"Input data must have {self.layers[0]} features, but got {len(x)}"

        ######
        # Iterate over layers
        ######
        for l in range(len(self.layers)-1):
            # Handle the bias Augmentation and structuring
            if l == 0:
                # First Layer is determinstic Special Case
                x = np.concatenate((x, np.ones((1, 1))), axis=0)
                # Blow up x to be a batch of (verif_samples, layers[0]+1)
                x_batch = np.tile(x.squeeze(), (self.verif_samples, 1))
                samples_before = x_batch
            else:
                # Otherwise, this are the post activation samples from the layer before
                samples_before = self.post_activation_samples[l-1]

            ###### 
            # Iterate over neurons
            ######
            for i in range(self.layers[l+1]):
                # Compute Pre-Activation Samples
                self.pre_activation_samples[l][:,i]  = np.sum(samples_before * self.weight_samples[l][:, :, i], axis=1)
                 
                # Compute the empirical moments of the sample set
                moments = np.zeros(self.moments_pre)
                for order in range(1, self.moments_pre+1):
                    moments[order-1] = np.mean(self.pre_activation_samples[l][:,i]**order)
                
                self.pre_activation_moments_samples[l][i, :] = moments

                # Compute Post Activation Samples
                self.post_activation_samples[l][:, i]= self.activation_functions[l](self.pre_activation_samples[l][:,i])

                # Compute the empirical moments of the sample set
                moments = np.zeros(self.moments_post)
                for order in range(1, self.moments_post+1):
                    moments[order-1] = np.mean(self.post_activation_samples[l][:, i]**order)

                self.post_activation_moments_samples[l][i, :] = moments

        ######
        return self.post_activation_samples[-1]
    
    def forward_combined(self,x):
        """
        Forward pass through the network
        Side by side version of different methods
        x: Input data, deterministic
        """
        #******Control******
        self.compute_moments = True
        self.compute_samples = True

        #******Preparation******
        if self.compute_samples:
            self.sample_weights()

        #******Handle and asseret input******
        # Handle scalar input in case it is given as a float or int
        if isinstance(x,float) or isinstance(x,int):    
            x = np.array([[x]])

        assert len(x) == self.layers[0], f"Input data must have {self.layers[0]} features, but got {len(x)}"

        ########
        # Loop over the layers
        ########
        for l in range(len(self.layers)-1):
            #******Moment-Based Handling******
            if self.compute_moments:
                # Handle the bias Augmentation and structuring
                if l == 0:
                    # First Layer is determinstic Special Case
                    x_moments = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
                    z_complete = np.stack((x_moments, np.zeros((x_moments.shape[0], x_moments.shape[1])), np.ones((x_moments.shape[0], x_moments.shape[1]))), axis=1) # Emulate a GM behsavior with mean=det, var=0 and weight=1
                    print("Output GM for the first layer")
                    print(z_complete)

                else:
                    # Attention: The bias is represented as a 1-component GM. The other entries have probably more
                    z_complete = np.stack((self.means_gm_post[l-1][:, :], self.variances_gm_post[l-1][:, :], self.weights_gm_post[l-1][:, :]),axis=1)
                    z_complete = np.concatenate((z_complete, np.zeros((1, 3, self.components_post))), axis=0)
                    z_complete[-1, 0, 0] = 1 # Bias as fictive GM component with mean 1
                    z_complete[-1, 1, 0] = 0 # and variance 0
                    z_complete[-1, 2, 0] = 1 # and weight 1
                    
                    print(f"Output GM for the {l}. layer")
                    print(z_complete)

            #******Sample-Based Handling******
            if self.compute_samples:
                # Handle the bias Augmentation and structuring
                if l == 0:
                    # First Layer is determinstic Special Case
                    x_samples = np.concatenate((x, np.ones((1, 1))), axis=0)
                    # Blow up x to be a batch of (verif_samples, layers[0]+1)
                    x_batch = np.tile(x_samples.squeeze(), (self.verif_samples, 1))
                    samples_before = x_batch
                else:
                    # Otherwise, this are the post activation samples from the layer before
                    samples_before = self.post_activation_samples[l-1]

            ########
            # Lop over the neurons in the layer
            ########
            for i in range(self.layers[l+1]):
                print(f"Layer {l}, Neuron {i} of {self.layers[l+1]}")
                # ******Moment-Based Pre-Activation Handling******
                if self.compute_moments:
                    # PRE ACTIVATION
                    # Assemble the weight array for the current neuron (Mean and Variance)
                    w_array = np.stack((self.weight_means[l][:,i], self.weight_variances[l][:,i]), axis=1)
                    # Compute Pre-Activation moments
                    print("Compute Pre-Activation Moments...")
                    moments_pre = moments_pre_act_combined_general(z_complete,w_array,order=self.moments_pre)
                    print("Done.")
                    self.pre_activation_moments_analytic[l][i,:] = moments_pre
                    # Match the Pre-Activation Moments to a GM
                    means, variances, weights = match_moments(moments_pre, components = self.components_pre)
                    self.means_gm_pre[l][i,:] = means
                    self.variances_gm_pre[l][i,:] = variances
                    self.weights_gm_pre[l][i,:] = weights
                    # Compute the fitted moments from the GM (as a backup)
                    for tt in range(self.moments_pre):
                        self.pre_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)
                    
                    # POST ACTIVATION
                    if self.activations[l] == 'relu':
                        # Compute moments of the post activation
                        moments_post =  moments_post_act(self.a_relu, self.means_gm_pre[l][i,:], self.variances_gm_pre[l][i,:], self.weights_gm_pre[l][i,:],order=self.moments_post)
                        self.post_activation_moments_analytic[l][i,:] = moments_post

                        # Matching with Peak of absorbed mass
                        if self.peak:                        
                            # If we set peak, we need to match the moments with a Dirac component at 0
                            absorbed_mass = self.compute_absorbed_mass(self.weights_gm_pre[l][i,:], self.means_gm_pre[l][i,:], self.variances_gm_pre[l][i,:])
                            means, variances, weights = match_moments_peak(moments_post, absorbed_mass, components = self.components_post)
                            self.means_gm_post[l][i,:] = means
                            self.variances_gm_post[l][i,:] = variances
                            self.weights_gm_post[l][i,:] = weights
                            self.dirac_weight_post[l][i] = absorbed_mass  # Set the Dirac weight to the absorbed mass      
                        
                            # Compute the fitted moments from the GM (as a backup)
                            for tt in range(self.moments_post):
                                self.post_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)
                        
                        # Conventional matching, no peak
                        else:                        
                            # Match the GM parameters to the moments
                            means, variances, weights = match_moments(moments_post, components = self.components_post)
                            self.means_gm_post[l][i,:] = means
                            self.variances_gm_post[l][i,:] = variances
                            self.weights_gm_post[l][i,:] = weights

                            # Compute the fitted moments from the GM (as a backup)
                            for tt in range(self.moments_post):
                                self.post_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)
                        

                    elif self.activations[l] == 'linear':
                        # Just take the pre moments as the transformation is linear with slope 1
                        moments_post = moments_pre
                        self.post_activation_moments_analytic[l][i,:] = moments_post   

                        # Match the GM parameters to the moments
                        means, variances, weights = match_moments(moments_post, components = self.components_post)
                        self.means_gm_post[l][i,:] = means
                        self.variances_gm_post[l][i,:] = variances
                        self.weights_gm_post[l][i,:] = weights

                        # Compute the fitted moments from the GM (as a backup)
                        for tt in range(self.moments_post):
                            self.post_activation_moments_fitted[l][i,tt] = gm_noncentral_moment(tt+1, weights, means, variances)

                    else:
                        raise ValueError(f"Unsupported activation function: {self.activations[l]} but should be {'relu'} or {'linear'}")

                # ******Sample-Based Pre-Activation Handling******
                if self.compute_samples:
                    # Compute Pre-Activation Samples
                    self.pre_activation_samples[l][:,i]  = np.sum(samples_before * self.weight_samples[l][:, :, i], axis=1)
                    
                    # Compute the empirical moments of the sample set
                    moments = np.zeros(self.moments_pre)
                    for order in range(1, self.moments_pre+1):
                        moments[order-1] = np.mean(self.pre_activation_samples[l][:,i]**order)
                    
                    self.pre_activation_moments_samples[l][i, :] = moments

                    # Compute Post Activation Samples
                    self.post_activation_samples[l][:, i]= self.activation_functions[l](self.pre_activation_samples[l][:,i])

                    # Compute the empirical moments of the sample set
                    moments = np.zeros(self.moments_post)
                    for order in range(1, self.moments_post+1):
                        moments[order-1] = np.mean(self.post_activation_samples[l][:, i]**order)

                    self.post_activation_moments_samples[l][i, :] = moments

                #******Compare Moments******
                if self.compute_moments and self.compute_samples:
                    # Compare the moments of the sample-based and moment-based approach
                    rel_pre = np.round(100 * (self.pre_activation_moments_analytic[l][i, :] - self.pre_activation_moments_samples[l][i, :]) / (self.pre_activation_moments_samples[l][i, :] + 1e-12), 2)
                    rel_post = np.round(100 * (self.post_activation_moments_analytic[l][i, :] - self.post_activation_moments_samples[l][i, :]) / (self.post_activation_moments_samples[l][i, :] + 1e-12), 2)

                    print(f"  Pre-Activation Relative Error: {rel_pre}")
                    print(f"  Post-Activation Relative Error: {rel_post}")
        return None