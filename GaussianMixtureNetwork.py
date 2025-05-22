#######################################################################################
# Utilities for Gaussian Mixture Network model 
# Author: Leon Winheim, ISAS at KIT Karlsruhe
# Date: 19.05.2025
#######################################################################################
import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import least_squares, LinearConstraint, minimize, Bounds
from math import factorial
import time

# Noncentral moments of a Gaussian, parametrized with the covariance
def  e1(mu):
    return mu
def  e2(mu,c_v):
    return c_v + mu**2
def  e3(mu,c_v):
    return 3*c_v*mu+mu**3
def  e4(mu,c_v):
    return 3*c_v**2 + 6*c_v*mu**2 + mu**4
def  e5(mu,c_v):
    return 15*c_v**2*mu + 10*c_v*mu**3 + mu**5
def e6(mu, c_v):
    return 15*c_v**3 + 45*c_v**2*mu**2 + 15*c_v*mu**4 + mu**6
def e7(mu, c_v):
    return 105*mu*(c_v**3)+105*(c_v**2)*(mu**3)+21*(mu**5)*c_v+mu**7
def e8(mu, c_v):    
    return 105*c_v**4+420*(c_v**3)*(mu**2)+210*(c_v**2)*(mu**4)+28*(c_v)*(mu**6)+mu**8
def e9(mu, c_v):
    return 945*(c_v**4)*mu+1260*(c_v**3)*(mu**3)+378*(c_v**2)*(mu**5)+36*c_v*(mu**7)+mu**9
def e10(mu, c_v):
    return 945*c_v**5+4725*(c_v**4)*(mu**2)+3150*(c_v**3)*(mu**4)+630*(c_v**2)*(mu**6)+45*(c_v)*mu**8+mu**10
def e11(mu, c_v):
    return 10395*c_v**5*mu + 17325*c_v**4*mu**3 + 6930*c_v**3*mu**5 + 990*c_v**2*mu**7 + 55*c_v*mu**9 + mu**11
def e12(mu, c_v):
    return 10395*c_v**6 + 62370*c_v**5*mu**2 + 51975*c_v**4*mu**4 + 13860*c_v**3*mu**6 + 1485*c_v**2*mu**8 + 66*c_v*mu**10 + mu**12
def e13(mu, c_v):
    return 135135*c_v**6*mu + 270270*c_v**5*mu**3 + 135135*c_v**4*mu**5 + 25740*c_v**3*mu**7 + 2145*c_v**2*mu**9 + 78*c_v*mu**11 + mu**13
def e14(mu, c_v):
    return 135135*c_v**7 + 945945*c_v**6*mu**2 + 945945*c_v**5*mu**4 + 315315*c_v**4*mu**6 + 45045*c_v**3*mu**8 + 3003*c_v**2*mu**10 + 91*c_v*mu**12 + mu**14
def e15(mu, c_v):
    return 2027025*c_v**7*mu + 4729725*c_v**6*mu**3 + 2837835*c_v**5*mu**5 + 675675*c_v**4*mu**7 + 75075*c_v**3*mu**9 + 4095*c_v**2*mu**11 + 105*c_v*mu**13 + mu**15
def e16(mu, c_v):
    return 2027025*c_v**8 + 16216200*c_v**7*mu**2 + 18918900*c_v**6*mu**4 + 7567560*c_v**5*mu**6 + 1351350*c_v**4*mu**8 + 120120*c_v**3*mu**10 + 5460*c_v**2*mu**12 + 120*c_v*mu**14 + mu**16
def e17(mu, c_v):
    return 34459425*c_v**8*mu + 91891800*c_v**7*mu**3 + 64324260*c_v**6*mu**5 + 18378360*c_v**5*mu**7 + 2552550*c_v**4*mu**9 + 185640*c_v**3*mu**11 + 7140*c_v**2*mu**13 + 136*c_v*mu**15 + mu**17
def e18(mu, c_v):
    return 34459425*c_v**9 + 310134825*c_v**8*mu**2 + 413513100*c_v**7*mu**4 + 192972780*c_v**6*mu**6 + 41351310*c_v**5*mu**8 + 4594590*c_v**4*mu**10 + 278460*c_v**3*mu**12 + 9180*c_v**2*mu**14 + 153*c_v*mu**16 + mu**18
def e19(mu, c_v):
    return 654729075*c_v**9*mu + 1964187225*c_v**8*mu**3 + 1571349780*c_v**7*mu**5 + 523783260*c_v**6*mu**7 + 87297210*c_v**5*mu**9 + 7936110*c_v**4*mu**11 + 406980*c_v**3*mu**13 + 11628*c_v**2*mu**15 + 171*c_v*mu**17 + mu**19
def e20(mu, c_v):
    return 654729075*c_v**10 + 6547290750*c_v**9*mu**2 + 9820936125*c_v**8*mu**4 + 5237832600*c_v**7*mu**6 + 1309458150*c_v**6*mu**8 + 174594420*c_v**5*mu**10 + 13226850*c_v**4*mu**12 + 581400*c_v**3*mu**14 + 14535*c_v**2*mu**16 + 190*c_v*mu**18 + mu**20

# Noncentral moments of a Gaussian Mixture for arbitrary many components
def e1_gm(w: np.array, mu: np.array, c: np.array):
    e1_gm = 0
    for i in range(len(w)):
        e1_gm += w[i] * e1(mu[i])
    return e1_gm

def e2_gm(w: np.array, mu: np.array, c: np.array):
    e2_gm = 0
    for i in range(len(w)):
        e2_gm += w[i] * e2(mu[i], c[i])
    return e2_gm

def e3_gm(w: np.array, mu: np.array, c: np.array):
    e3_gm = 0
    for i in range(len(w)):
        e3_gm += w[i] * e3(mu[i], c[i])
    return e3_gm

def e4_gm(w: np.array, mu: np.array, c: np.array):
    e4_gm = 0
    for i in range(len(w)):
        e4_gm += w[i] * e4(mu[i], c[i])
    return e4_gm

def e5_gm(w: np.array, mu: np.array, c: np.array):
    e5_gm = 0
    for i in range(len(w)):
        e5_gm += w[i] * e5(mu[i], c[i])
    return e5_gm

def e6_gm(w: np.array, mu: np.array, c: np.array):
    e6_gm = 0
    for i in range(len(w)):
        e6_gm += w[i] * e6(mu[i], c[i])
    return e6_gm

def e7_gm(w: np.array, mu: np.array, c: np.array):
    e7_gm = 0
    for i in range(len(w)):
        e7_gm += w[i] * e7(mu[i], c[i])
    return e7_gm

def e8_gm(w: np.array, mu: np.array, c: np.array):
    e8_gm = 0
    for i in range(len(w)):
        e8_gm += w[i] * e8(mu[i], c[i])
    return e8_gm

def e9_gm(w: np.array, mu: np.array, c: np.array):
    e9_gm = 0
    for i in range(len(w)):
        e9_gm += w[i] * e9(mu[i], c[i])
    return e9_gm

def e10_gm(w: np.array, mu: np.array, c: np.array):
    e10_gm = 0
    for i in range(len(w)):
        e10_gm += w[i] * e10(mu[i], c[i])
    return e10_gm

def e11_gm(w: np.array, mu: np.array, c: np.array):
    e11_gm = 0
    for i in range(len(w)):
        e11_gm += w[i] * e11(mu[i], c[i])
    return e11_gm

def e12_gm(w: np.array, mu: np.array, c: np.array):
    e12_gm = 0
    for i in range(len(w)):
        e12_gm += w[i] * e12(mu[i], c[i])
    return e12_gm

def e13_gm(w: np.array, mu: np.array, c: np.array):
    e13_gm = 0
    for i in range(len(w)):
        e13_gm += w[i] * e13(mu[i], c[i])
    return e13_gm

def e14_gm(w: np.array, mu: np.array, c: np.array):
    e14_gm = 0
    for i in range(len(w)):
        e14_gm += w[i] * e14(mu[i], c[i])
    return e14_gm

def e15_gm(w: np.array, mu: np.array, c: np.array):
    e15_gm = 0
    for i in range(len(w)):
        e15_gm += w[i] * e15(mu[i], c[i])
    return e15_gm

def e16_gm(w: np.array, mu: np.array, c: np.array):
    e16_gm = 0
    for i in range(len(w)):
        e16_gm += w[i] * e16(mu[i], c[i])
    return e16_gm

def e17_gm(w: np.array, mu: np.array, c: np.array):
    e17_gm = 0
    for i in range(len(w)):
        e17_gm += w[i] * e17(mu[i], c[i])
    return e17_gm

def e18_gm(w: np.array, mu: np.array, c: np.array):
    e18_gm = 0
    for i in range(len(w)):
        e18_gm += w[i] * e18(mu[i], c[i])
    return e18_gm

def e19_gm(w: np.array, mu: np.array, c: np.array):
    e19_gm = 0
    for i in range(len(w)):
        e19_gm += w[i] * e19(mu[i], c[i])
    return e19_gm

def e20_gm(w: np.array, mu: np.array, c: np.array):
    e20_gm = 0
    for i in range(len(w)):
        e20_gm += w[i] * e20(mu[i], c[i])
    return e20_gm

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

def moments_pre_act_combined_general(z:np.ndarray,w:np.ndarray):
    """ This function computes the first ten moments of the pre activation value for a single neuron with multiple products input and weight.

        z is an array containing the the input values. Every row is a different GM with [[mu1,mu2...muN],[c1,c2...cN],[w1,w2...wN]] and resembles one neuronal output from before
        w is an array containing tuples of the form (mu_w, c_w) for every entry representing the weight as independent Gaussian

        returns an array of the first ten moments of the pre activation value
    """

    number_moments = 10 #If we want to change this, we need to add more moments to the functions above

    # In general, x_array could be a gaussian mixture or deterministic. 
    # If it is deterministic, it will be the value as the first mean, 1 as the first weight and all other values in the GM specification will be zero
    
    assert z.shape[0] == w.shape[0], "z and w must have the same length, but have lengths {} and {}".format(z.shape[0], w.shape[0])  

    # Pre-Compute the moments of each w (w is always a Gaussian, so this is always needed)
    moments_w = np.zeros((len(w), number_moments+1))
    for i in range(w.shape[0]):
        mu_w, c_w = w[i]
        # Save the moments, augment a 0 moment to avoid indexing issues with the multinomial indexing
        moments_w[i] = [1 ,e1(mu_w), e2(mu_w, c_w), e3(mu_w, c_w), e4(mu_w, c_w), e5(mu_w, c_w), e6(mu_w, c_w), e7(mu_w, c_w), e8(mu_w, c_w), e9(mu_w, c_w), e10(mu_w, c_w)]

    # Pre-Compute the Moments of each z (z could be Deterministic or GM, but the gm-moment-generation will handle this flexible)
    moments_z = np.zeros((len(z), number_moments+1))
    for i in range(z.shape[0]):
        # mu, c and w are potentially multi-component GMs
        mu_z, c_z, w_z = z[i]
        # Save the moments, augment a 0 moment to avoid indexing issues with the multinomial indexing
        moments_z[i] = [1 ,e1_gm(w_z, mu_z, c_z), e2_gm(w_z, mu_z, c_z), e3_gm(w_z, mu_z, c_z), e4_gm(w_z, mu_z, c_z), e5_gm(w_z, mu_z, c_z), e6_gm(w_z, mu_z, c_z), e7_gm(w_z, mu_z, c_z), e8_gm(w_z, mu_z, c_z), e9_gm(w_z, mu_z, c_z), e10_gm(w_z, mu_z, c_z)]

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

def moments_post_act(a:float,mu:np.ndarray,c:np.ndarray,w:np.ndarray):
    """This function computes the post activation moments of a Gaussian mixture with arbitrary many components propagated through leaky relu
    
        a: slope of the leaky relu
        mu: mean vector of the Gaussian Mixture
        c: variance array of the Gaussian Mixture
        w: weights of the Gaussian Mixture

        returns the first ten moments of the post activation distribution
    """
    assert np.max(c) >= 0.01, "Variance must be big enough"
    num_moments = 10
    assert len(mu) == len(c) == len(w), "mu, c, and w must have the same length"
    sigma = np.sqrt(c)

    # Initialize result vector 
    result = np.zeros(num_moments, dtype=float)

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
        
    return result

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
    params[components] = -1.8    
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
                            options={'disp': False, 'xtol': 1e-6, 'gtol': 1e-6, 'maxiter': 1000}
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
    if components != 2:
        raise ValueError("This function only works for two components")
    
    #params = [0.5,0.0,5.0,1.0,1.0]  # Initial guess for [w0, mu0, mu1, c0, c1]
    params = [0.5,0.0,1.0,0.1,0.1]  # Initial guess for [w0, mu0, mu1, c0, c1]

    # Call optimizer with bounds
    start = time.time()
    result = least_squares(residuals_matching_2, params, args=in_mom, bounds=([0, -np.inf, -np.inf, 0, 0], [1.0, np.inf, np.inf, np.inf, np.inf]))
    end = time.time()
    #print("Time for optimization: {}".format(end-start))

    w_res = [result.x[0], 1 - result.x[0]]
    mu_res = [result.x[1], result.x[2]]
    c_res = [result.x[3], result.x[4]]

    return np.array(mu_res), np.array(c_res), np.array(w_res)

def match_moments_special(in_mom,components,solver="trust-constr"):
    """Function to perform the moment matching for given Moments and a desired number of components.
        Solver can be either SLSQP or trust-constr.
        This  fixes some dirac like GM component at 0
    """

    #Fallback
    if components == 2:
        return match_moments_2_special(in_mom, components)

    else:
        raise ValueError("This function only works for two components yet")
    # Assemble the parameter vector according to the number of components. Per Component, we have 3 parameters: w, mu, c
    # The complete parameter vector will look like this: [w0,w1...,wN,mu0,mu1,...,muN,c0,c1,...,cN]
    params = np.zeros(components*3, dtype=float)

    #Set the initial guess for the weights as equal and summing up to one
    for i in range(components):
        params[i] = 1/components

    #Set the initial guess for the means as the input moments
    params[components] = -1.8    
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
                            options={'disp': False, 'xtol': 1e-6, 'gtol': 1e-6, 'maxiter': 1000}
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

def match_moments_2_special(in_mom, components):
    """Intermediate solution, only for two components"""
    if components != 2:
        raise ValueError("This function only works for two components")
    
    #params = [0.5,0.0,5.0,1.0,1.0]  # Initial guess for [w0, mu0, mu1, c0, c1]
    params = [0.5, 1.0, 0.01, 9.1]  # Initial guess for [w0, mu1, c0, c1]

    # Call optimizer with bounds
    start = time.time()
    result = least_squares(residuals_matching_2_special, params, args=in_mom, bounds=([0, -np.inf, 0, 0], [1.0, np.inf, np.inf, np.inf]))
    end = time.time()
    #print("Time for optimization: {}".format(end-start))

    w_res = [result.x[0], 1 - result.x[0]]
    mu_res = [0.0, result.x[1]]
    c_res = [result.x[2], result.x[3]]

    return np.array(mu_res), np.array(c_res), np.array(w_res)


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

    # Infer how many components we have
    components = int(len(params)/3)

    # Extract the parameters from the input vector
    w = params[:components]
    mu = params[components:components*2]
    c = params[components*2:components*3]

    # Compute the moments of the Gaussian Mixture
    gm_moments = np.zeros(10, dtype=float)
    gm_moments[0] = e1_gm(w, mu, c)
    gm_moments[1] = e2_gm(w, mu, c)
    gm_moments[2] = e3_gm(w, mu, c)
    gm_moments[3] = e4_gm(w, mu, c)
    gm_moments[4] = e5_gm(w, mu, c)
    gm_moments[5] = e6_gm(w, mu, c)
    gm_moments[6] = e7_gm(w, mu, c)
    gm_moments[7] = e8_gm(w, mu, c)
    gm_moments[8] = e9_gm(w, mu, c)
    gm_moments[9] = e10_gm(w, mu, c)

    # Compute the weighted residuals
    residuals = np.zeros(10, dtype=float)
    for i in range(10):
        residuals[i] = abs(gm_moments[i] - t[i])/t[i]
    
    # COmpute the summed squared residuals
    residuals = np.sum(residuals**2)

    # This should return an array
    return residuals

def residuals_matching_2(params, *args):
    """
        Compute the residual value for the optimization process.
        Returns the array of indivudal residuals for each moment.
    """
    # Unpack the arguments
    t = []
    for temp in args:
        t.append(temp)

    # Infer how many components we have
    components = int(len(params)/3)
    # Extract the parameters from the input vector
    w = np.array([params[:1].squeeze(),1-params[:1].squeeze()])
    mu = params[1:3]
    c = params[3:5]

    # Compute the moments of the Gaussian Mixture
    gm_moments = np.zeros(10, dtype=float)
    gm_moments[0] = e1_gm(w, mu, c)
    gm_moments[1] = e2_gm(w, mu, c)
    gm_moments[2] = e3_gm(w, mu, c)
    gm_moments[3] = e4_gm(w, mu, c)
    gm_moments[4] = e5_gm(w, mu, c)
    gm_moments[5] = e6_gm(w, mu, c)
    gm_moments[6] = e7_gm(w, mu, c)
    gm_moments[7] = e8_gm(w, mu, c)
    gm_moments[8] = e9_gm(w, mu, c)
    gm_moments[9] = e10_gm(w, mu, c)

    # Compute the weighted residuals
    residuals = np.zeros(10, dtype=float)
    for i in range(10):
        residuals[i] = abs(gm_moments[i] - t[i])/t[i]

    # This should return an array
    return residuals

def residuals_matching_2_special(params, *args):
    """
        Compute the residual value for the optimization process.
        Returns the array of indivudal residuals for each moment.
    """
    # Unpack the arguments [w0, mu1, c0, c1]
    t = []
    for temp in args:
        t.append(temp)
    t = np.array(t).squeeze()	

    # Infer how many components we have
    components = int(len(params)/3)
    # Extract the parameters from the input vector
    w = np.array([params[0].squeeze(),1-params[0].squeeze()])
    mu = np.array([0.0, params[1].squeeze()])
    c = np.array([params[2].squeeze(), params[3].squeeze()])

    # Compute the moments of the Gaussian Mixture
    gm_moments = np.zeros(10, dtype=float)
    gm_moments[0] = e1_gm(w, mu, c)
    gm_moments[1] = e2_gm(w, mu, c)
    gm_moments[2] = e3_gm(w, mu, c)
    gm_moments[3] = e4_gm(w, mu, c)
    gm_moments[4] = e5_gm(w, mu, c)
    gm_moments[5] = e6_gm(w, mu, c)
    gm_moments[6] = e7_gm(w, mu, c)
    gm_moments[7] = e8_gm(w, mu, c)
    gm_moments[8] = e9_gm(w, mu, c)
    gm_moments[9] = e10_gm(w, mu, c)

    # Compute the weighted residuals
    residuals = np.zeros(10, dtype=float)
    for i in range(10):
        residuals[i] = abs(gm_moments[i] - t[i])/t[i]

    # This should return an array
    return residuals

#########################################################################################   
# Define the network class to build actual architecture

class GaussianMixtureNetwork():
    """
    Class to build a Gaussian Mixture based BNN. Numpy-Based
    """
    def __init__(self,layers:list,activations:list,components_pre:int,components_post:int,a_relu:float=0.01):
        """
        Initialize the Gaussian Mixture Network with the given parameters.

        layers: List of integers representing the number of neurons in each layer. First is Input feautres, last is output features
        activations: List of activation functions for each layer. Contains one entry less than layers
        components_pre: Number of components for the pre-activation layer.
        components_post: Number of components for the post-activation layer.
        """
        # Add values to instance
        self.layers = layers
        self.activations = activations.copy()
        self.components_pre = components_pre
        self.components_post = components_post
        self.a_relu = a_relu
        self.verif_samples = 100000

        # Intialize the weights and biases
        self.init_parameters()

        # Reset actiuvation functions
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

        # Initialize the containers for the GM parameters in pre-activation
        self.means_gm_pre = []
        self.variances_gm_pre = []
        self.weights_gm_pre = []
        # Iterate over the layers
        for i in range(len(self.layers)-1):
            # Append empty lists for each layer
            self.means_gm_pre.append(np.zeros((self.layers[i+1], self.components_pre)))
            self.variances_gm_pre.append(np.zeros((self.layers[i+1], self.components_pre)))
            self.weights_gm_pre.append(np.zeros((self.layers[i+1], self.components_pre)))

        # Initialize the containers for the GM parameters in post-activation
        self.means_gm_post = []
        self.variances_gm_post = []
        self.weights_gm_post = []
        # Iterate over the layers
        for i in range(len(self.layers)-1):
            # Append empty lists for each layer
            self.means_gm_post.append(np.zeros((self.layers[i+1], self.components_post)))
            self.variances_gm_post.append(np.zeros((self.layers[i+1], self.components_post)))
            self.weights_gm_post.append(np.zeros((self.layers[i+1], self.components_post)))

        # Generate pre-activation moment container for analytic
        self.pre_activation_moments_analytic = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],10))

            self.pre_activation_moments_analytic.append(samples)
        
        # Generate post-activation moment container for analytic
        self.post_activation_moments_analytic = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],10))

            self.post_activation_moments_analytic.append(samples)

    def sample_weights(self):
        """Generate samples of every weight and bias in the network"""
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
            samples = np.zeros((self.layers[i+1],10))

            self.pre_activation_moments_samples.append(samples)
        
        # Generate post-activation moment container
        self.post_activation_moments_samples = []
        # Iterate through layers
        for i in range(len(self.layers)-1):
            samples = np.zeros((self.layers[i+1],10))

            self.post_activation_moments_samples.append(samples)
        
    def set_act_func(self):
        """Set the activation functions for the network"""
        # Initialize the activation functions
        self.activation_functions = []
        for i in range(len(self.layers)-1):
            if self.activations[i] == 'relu':
                self.activation_functions.append(self.relu)
            elif self.activations[i] == 'linear':
                self.activation_functions.append(self.linear)
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]} but should be {'relu'}")
            
    def relu(self, x, return_name=False):
        """General ReLU activation function with leaky parameter"""
        if return_name:
            return "relu"
        a = self.a_relu  
        return np.where(x > 0, x, a * x)
    
    def linear(self, x, return_name=False):  
        """Linear activation function"""
        if return_name:
            return "linear"
        return x

    def print_network(self):
        """
        Print out the network structure with activation functions.
        """
        print("Gaussian Mixture Network Structure:")
        for i in range(len(self.layers) - 1):
            act = self.activations[i] if i < len(self.activations) else "none"
            print(f" Layer {i}: {self.layers[i]} -> {self.layers[i+1]} neurons | Activation: {act}")
        print(f" Pre-activation GM components: {self.components_pre}")
        print(f" Post-activation GM components: {self.components_post}")
        print(f" Leaky ReLU slope (a): {self.a_relu}")

    def compare_sample_moments_forward(self,x):
        """Makes a forward pass in both implemented methods and compares the moments"""
        print("VERIFICATION")
        print()
        start = time.time()
        result = self.forward_samples(x)
        stop = time.time()
        print(f"Time for forward_samples: {stop-start:.2f} s")

        start = time.time()
        result = self.forward_moments(x)
        stop = time.time()
        print(f"Time for momentmatch: {stop-start:.2f} s")

        for i in range(len(self.pre_activation_moments_analytic)):
            rel_pre = np.round(100*(self.pre_activation_moments_analytic[i]-self.pre_activation_moments_samples[i])/self.pre_activation_moments_samples[i],2)

            rel_post = np.round(100*(self.post_activation_moments_analytic[i]-self.post_activation_moments_samples[i])/self.post_activation_moments_samples[i],2)

            print()
            print(f"******Layer {i}******")
            print(f"Max. Pre activation rel. error: {np.max(abs(rel_pre))} %; Max. First Moment rel. error: {np.max(abs(rel_pre[:,0]))} %")
            print(f"Max. Post activation rel. error: {np.max(abs(rel_post))} %; Max. First Moment rel. error: {np.max(abs(rel_post[:,0]))} %")

    def forward_moments(self,x):
        """
        Forward pass through the network based on the method of moments.
        x: Input data, deterministic
        """
        if isinstance(x,float) or isinstance(x,int):    
            x = np.array([[x]])

        assert len(x) == self.layers[0], f"Input data must have {self.layers[0]} features, but got {len(x)}"

        # Augment Bias 
        x = np.concatenate((x, np.ones((1, 1))), axis=0)

        # Indexing a GM Parameter value is done by [layer][neuron,component]

        ######
        # Handle the first layer seperately as it uses deterministic input
        ######
        # Note: I assume that thr first ectivation is ReLU
        # Iterate over the neurons in the first layer
        for i in range(self.layers[1]):
            # Match Pre-Activation
            # Compute moments of the pre activation (Weights for one neuron are one column of the weight matrix)
            w_array = np.stack((self.weight_means[0][:,i], self.weight_variances[0][:,i]), axis=1)
            x_array = np.stack((x,np.zeros((x.shape[0], 1)),np.ones((x.shape[0],1))), axis=1) # Emulate a GM behsavior with mean=det, var=0 and weight=1
            moments_pre = moments_pre_act_combined_general(x_array,w_array)
            
            self.pre_activation_moments_analytic[0][i,:] = moments_pre

            # Match the GM parameters to the moments
            means, variances, weights = match_moments(moments_pre, components = self.components_pre)

            self.means_gm_pre[0][i,:] = means
            self.variances_gm_pre[0][i,:] = variances
            self.weights_gm_pre[0][i,:] = weights

            # Match Post-Activation
            # Compute moments of the post activation
            moments_post =  moments_post_act(self.a_relu, means, variances, weights)

            self.post_activation_moments_analytic[0][i,:] = moments_post

            # Match the GM parameters to the moments
            means, variances, weights = match_moments(moments_post, components = self.components_post)

            self.means_gm_post[0][i,:] = means
            self.variances_gm_post[0][i,:] = variances
            self.weights_gm_post[0][i,:] = weights

        ######
        # Iterate over the rest of the layers
        # I need to handle the bias term seperately as it is (Gauss x Deterministic)
        ######
        for l in range(1, len(self.layers)-1):
            # Iterate over the neurons in the next layer
            for i in range(0,self.layers[l+1]):
                # Match Pre-Activation
                # Compute moments of the pre activation. This takes the weights as Gaussians and the previous layer output as GM, except for the bias neuron
                # Augment the Bias neuron and assemble the parameter list
                z_complete = np.stack((self.means_gm_post[l-1][:, :], self.variances_gm_post[l-1][:, :], self.weights_gm_post[l-1][:, :]),axis=1)
                z_complete = np.concatenate((z_complete, np.zeros((1, 3, self.components_post))), axis=0)
                z_complete[-1, 0, 0] = 1 # Bias as fictive GM component with mean 1, cov 0 
                z_complete[-1, 1, 0] = 0 # and variance 0
                z_complete[-1, 2, 0] = 1 # and weight 1
                w_complete = np.stack((self.weight_means[l][:,i], self.weight_variances[l][:,i]), axis=1)
                moments_pre = moments_pre_act_combined_general(z_complete, w_complete)

                self.pre_activation_moments_analytic[l][i,:] = moments_pre

                # Match the GM parameters to the moments
                means, variances, weights = match_moments(moments_pre, components = self.components_pre)

                self.means_gm_pre[l][i,:] = means
                self.variances_gm_pre[l][i,:] = variances
                self.weights_gm_pre[l][i,:] = weights

                # Match Post-Activation
                if self.activations[l] == 'relu':
                    # Compute moments of the post activation
                    moments_post =  moments_post_act(self.a_relu, means, variances, weights)

                    self.post_activation_moments_analytic[l][i,:] = moments_post

                    # Match the GM parameters to the moments
                    means, variances, weights = match_moments(moments_post, components = self.components_post)
                    self.means_gm_post[l][i,:] = means
                    self.variances_gm_post[l][i,:] = variances
                    self.weights_gm_post[l][i,:] = weights
                    
                elif self.activations[l] == 'linear':
                    moments_post = moments_pre

                    self.post_activation_moments_analytic[l][i,:] = moments_post    

                    # Match the GM parameters to the moments
                    means, variances, weights = match_moments(moments_post, components = self.components_post)
                    self.means_gm_post[l][i,:] = means
                    self.variances_gm_post[l][i,:] = variances
                    self.weights_gm_post[l][i,:] = weights

        ######
        return self.means_gm_post[-1], self.variances_gm_post[-1], self.weights_gm_post[-1]

    def forward_samples(self,x):
        """
        Forward pass through the network based on samples.
        x: Input data, deterministic
        """
        # Generate sample representationj of every weight and prepare the intermediate sample values
        self.sample_weights()

        if isinstance(x,float) or isinstance(x,int):    
            x = np.array([[x]])

        assert len(x) == self.layers[0], f"Input data must have {self.layers[0]} features, but got {len(x)}"

        # Augment Bias 
        x = np.concatenate((x, np.ones((1, 1))), axis=0)

        # Indexing a GM Parameter value is done by [layer][neuron,component]

        ######
        # Handle the first layer seperately as it uses deterministic input
        ######
        # Note: I assume that thr first ectivation is ReLU
        # Iterate over the neurons in the first layer
        for i in range(self.layers[1]):
            # Compute pre avtivation 
            # pre_act_samples shape: (verif_samples,)
            pre_act_samples = np.dot(x.squeeze(), self.weight_samples[0][:, :, i].T)
            
            # Store the samples in the pre activation sample container
            self.pre_activation_samples[0][:, i] = pre_act_samples

            # Compute the empirical moments of the sample set
            moments = np.zeros(10)
            for order in range(1, 11):
                moments[order-1] = np.mean(pre_act_samples**order)

            self.pre_activation_moments_samples[0][i, :] = moments

            post_act_samples = self.activation_functions[0](pre_act_samples)

            # Compute the empirical moments of the sample set
            moments = np.zeros(10)
            for order in range(1, 11):
                moments[order-1] = np.mean(post_act_samples**order)

            self.post_activation_moments_samples[0][i, :] = moments

            # Store the samples in the post activation sample container
            self.post_activation_samples[0][:, i] = post_act_samples


        ######
        # Iterate over the rest of the layers
        # I need to handle the bias term seperately as it is (Gauss x Deterministic)
        ######
        for l in range(1, len(self.layers)-1):
            # Compute pre avtivation 
            # pre_act_samples shape: (verif_samples,neurons)
            # Append the post activation samples with from before with bias samples
            pre_act_samples = np.einsum('bi,bij->bj', self.post_activation_samples[l-1], self.weight_samples[l])

            # Store the samples in the pre activation sample container
            self.pre_activation_samples[l][:,:] = pre_act_samples

            # Compute the empirical moments of the sample set
            for i in range(self.layers[l+1]):
                moments = np.zeros(10)
                for order in range(1, 11):
                    moments[order-1] = np.mean(pre_act_samples[:,i]**order)

                self.pre_activation_moments_samples[l][i, :] = moments

            post_act_samples = self.activation_functions[l](pre_act_samples)

            # Compute the empirical moments of the sample set
            for i in range(self.layers[l+1]):
                moments = np.zeros(10)
                for order in range(1, 11):
                    moments[order-1] = np.mean(post_act_samples[:,i]**order)

                self.post_activation_moments_samples[l][i, :] = moments


            self.post_activation_samples[l][:,:-1] = post_act_samples

        ######
        return self.post_activation_samples[-1]