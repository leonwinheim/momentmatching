###########################################################################
# Collect some functions to be shared in the GM BNN Project
# Author: Leon Winheim
# Date: 02.05.2025
###########################################################################
import numpy as np
from sympy.parsing.mathematica import parse_mathematica
from sympy import var, erf, erfc
import sklearn
import re
from scipy.optimize import least_squares, minimize

# Replace \[Pi] with Pi in the expressions so python can read it
def replace_pi(expression):
    if expression is None:
        return expression
    return expression.replace("\\[Pi]", "Pi")

# Replace mu[i] with mui when i is a variable integer
def replace_mu_index(expression):
    if expression is None:
        return expression
    # Use regex to find patterns like mu[i] where i is an integer
    pattern = r"mu\[(\d+)\]"
    # Replace mu[i] with mui
    return re.sub(pattern, lambda match: f"mu{match.group(1)}", expression)

# Replace w[i] with wi when i is a variable integer
def replace_w_index(expression):
    if expression is None:
        return expression
    # Use regex to find patterns like w[i] where i is an integer
    pattern = r"w\[(\d+)\]"
    # Replace w[i] with wi
    return re.sub(pattern, lambda match: f"w{match.group(1)}", expression)

def replace_c_index(expression):
    if expression is None:
        return expression
    # Use regex to find patterns like c[i] where i is an integer
    pattern = r"c\[(\d+)\]"
    # Replace c[i] with ci
    return re.sub(pattern, lambda match: f"c{match.group(1)}", expression)

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

# Noncentral moments of a Gaussian Mixture with two components
def e1_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e1(mu0_v) + w1_v * e1(mu1_v))

def e2_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e2(mu0_v, c0_v) + w1_v * e2(mu1_v, c1_v))

def e3_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e3(mu0_v, c0_v) + w1_v * e3(mu1_v, c1_v))

def e4_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e4(mu0_v, c0_v) + w1_v * e4(mu1_v, c1_v))

def e5_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e5(mu0_v, c0_v) + w1_v * e5(mu1_v, c1_v))

def e6_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e6(mu0_v, c0_v) + w1_v * e6(mu1_v, c1_v))

def e7_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e7(mu0_v, c0_v) + w1_v * e7(mu1_v, c1_v))

def e8_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e8(mu0_v, c0_v) + w1_v * e8(mu1_v, c1_v))

def e9_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e9(mu0_v, c0_v) + w1_v * e9(mu1_v, c1_v))

def e10_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e10(mu0_v, c0_v) + w1_v * e10(mu1_v, c1_v))

def e11_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e11(mu0_v, c0_v) + w1_v * e11(mu1_v, c1_v))

def e12_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e12(mu0_v, c0_v) + w1_v * e12(mu1_v, c1_v))

def e13_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e13(mu0_v, c0_v) + w1_v * e13(mu1_v, c1_v))

def e14_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e14(mu0_v, c0_v) + w1_v * e14(mu1_v, c1_v))

def e15_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e15(mu0_v, c0_v) + w1_v * e15(mu1_v, c1_v))

def e16_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e16(mu0_v, c0_v) + w1_v * e16(mu1_v, c1_v))

def e17_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e17(mu0_v, c0_v) + w1_v * e17(mu1_v, c1_v))

def e18_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e18(mu0_v, c0_v) + w1_v * e18(mu1_v, c1_v))

def e19_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e19(mu0_v, c0_v) + w1_v * e19(mu1_v, c1_v))

def e20_gm2(w0_v, w1_v, mu0_v, mu1_v, c0_v, c1_v):
    return (w0_v * e20(mu0_v, c0_v) + w1_v * e20(mu1_v, c1_v))

# Next generation of functions
def moments_pre_act_single_det(x,mu_w,c_w):
    """This function computes the first ten moments of the random variable product w*x, 
        where z_in is a dseterministic scalar valkue and w is a Gaussian random variable.

        x: The input value
        mu_w: Mean of the Gaussian random variable w
        c_w: Variance of the Gaussian random variable w

        Returns: A vector of the first ten moments of the random variable product w*x
        """
    # Check the shape of the input
    #assert x.shape == (1,), "x must be a scalar value"

    # Initialize result vector 
    result = np.zeros(10, dtype=float)

    result[0] = x*e1(mu_w)           #First Moment
    result[1] = x**2*e2(mu_w,c_w)    #Second Moment
    result[2] = x**3*e3(mu_w,c_w)    #Third Moment
    result[3] = x**4*e4(mu_w,c_w)    #Fourth Moment
    result[4] = x**5*e5(mu_w,c_w)    #Fifth Moment
    result[5] = x**6*e6(mu_w,c_w)    #Sixth Moment
    result[6] = x**7*e7(mu_w,c_w)    #Seventh Moment
    result[7] = x**8*e8(mu_w,c_w)    #Eighth Moment
    result[8] = x**9*e9(mu_w,c_w)    #Ninth Moment
    result[9] = x**10*e10(mu_w,c_w)  #Tenth Moment

    return result


def moments_pre_act_combined_det(x_list,w_list):
    """ This function computes the first ten moments of the pre activation value for a single neuron with multiple products input and weight.

        x_list is a list containing the input values
        w_list is a list containing tuples of the form (mu_w, c_w) for every entry repsresenting the weights as Gaussian

        returns a vector of the first ten moments of the pre activation value
    """

    #MISLEADING: BOTH INPUTS ARE ARRAYS

    number_moments = 10

    assert len(x_list) == len(w_list), "x_list and w_list must have the same length, but have lengths {} and {}".format(len(x_list), len(w_list))  

    # Initialize result vector 
    result = np.zeros(number_moments)

    # Iterate through the list
    for i in range(len(x_list)):
        x = x_list[i]
        mu_w, c_w = w_list[i]

        # Call the moments_pre_act_single function for each entry
        result += moments_pre_act_single_det(x,mu_w,c_w)

    return result


def moments_pre_act_single(mu_w:np.ndarray,c_w:np.ndarray,mu_z:np.ndarray,c_z:np.ndarray,w_z:np.ndarray):
    """This function computes the first ten moments of the random variable product w*z_in, 
        where z_in is a scalar Gaussian Mixture with arbitrary number of components and w is a Gaussian random variable.

        mu_w: Mean of the Gaussian random variable w
        c_w: Variance of the Gaussian random variable w
        mu_z: Mean vector of the Gaussian Mixture
        c_z: Variance array of the Gaussian Mixture
        w_z: Weights of the Gaussian Mixture

        Returns: A vector of the first ten moments of the random variable product w*z_in
        """
    
    num_moments = 10
    # Check ths shape parameters of the GM
    assert len(mu_z) == len(c_z) == len(w_z), "mu_z, c_z, and w_z must have the same length"
    #assert len(mu_w) == len(c_w) == 1 , "mu_w and c_w must have length 1"
    assert np.isclose(np.sum(w_z), 1.0), "Weights of the Gaussian Mixture must sum to 1"

    # Initialize result vector 
    result = np.zeros(num_moments, dtype=float)

    #Iterate over the components of the GM
    for i in range(len(mu_z)):
        assert c_z[i] >= 0, "Variance must be non-negative"
        # Add moments up to ten for each component
        result[0] += w_z[i] * mu_w * mu_z[i]    #First Moment
        result[1] += w_z[i] * (mu_w**2 + c_w) * (c_z[i] + mu_z[i]**2)    #Second Moment
        result[2] += w_z[i] * mu_w *mu_z[i]*(mu_w**2+3*c_w)*(mu_z[i]**2+3*c_z[i])    #Third Moment
        result[3] += w_z[i] * (mu_w**4+6*mu_w**2*c_w+3*c_w**2)*(mu_z[i]**4+6*mu_z[i]**2*c_z[i]+3*c_z[i]**2)  #Fourth Moment
        result[4] += w_z[i] * mu_w*mu_z[i]*(mu_w**4 +10*mu_w**2*c_w+15*c_w**2)*(mu_z[i]**4+10*mu_z[i]**2*c_z[i]+15*c_z[i]**2)  #Fifth Moment
        result[5] += w_z[i] * (mu_w**6+15*mu_w**4*c_w+45*mu_w**2*c_w**2+15*c_w**3)*(mu_z[i]**6+15*mu_z[i]**4*c_z[i]+45*mu_z[i]**2*c_z[i]**2+15*c_z[i]**3)  #Sixth Moment
        result[6] += w_z[i] * mu_w*mu_z[i]*(mu_w**6+21*mu_w**4*c_w+105*mu_w**2*c_w**2+105*c_w**3)*(mu_z[i]**6+21*mu_z[i]**4*c_z[i]+105*mu_z[i]**2*c_z[i]**2+105*c_z[i]**3)  #Seventh Moment
        result[7] += w_z[i] * (mu_w**8+28*mu_w**6*c_w+210*mu_w**4*c_w**2+420*mu_w**2*c_w**3+28*c_w**4)*(mu_z[i]**8+28*mu_z[i]**6*c_z[i]+210*mu_z[i]**4*c_z[i]**2+420*mu_z[i]**2*c_z[i]**3+28*c_z[i]**4)  #Eighth Moment
        result[8] += w_z[i] * mu_w*mu_z[i]*(mu_w**8+36*mu_w**6*c_w+378*mu_w**4*c_w**2+1260*mu_w**2*c_w**3+378*c_w**4)*(mu_z[i]**8+36*mu_z[i]**6*c_z[i]+378*mu_z[i]**4*c_z[i]**2+1260*mu_z[i]**2*c_z[i]**3+378*c_z[i]**4)  #Ninth Moment
        result[9] += w_z[i] * (mu_w**10+45*mu_w**8*c_w+630*mu_w**6*c_w**2+3150*mu_w**4*c_w**3+4725*mu_w**2*c_w**4+945*c_w**5)*(mu_z[i]**10+45*mu_z[i]**8*c_z[i]+630*mu_z[i]**6*c_z[i]**2+3150*mu_z[i]**4*c_z[i]**3+4725*mu_z[i]**2*c_z[i]**4+945*c_z[i]**5)  #Tenth Moment

    return result

def moments_pre_act_combined(w_list,z_list):
    """ This function computes the first ten moments of the pre activation value for a single neuron with multiple products of random variables.

        w_list is a list containing tuples of the form (mu_w, c_w) for every entry repsresenting the weights as Gaussian
        z_list is a list containing tuples of the form ((mu_z1,mu_z2), (c_z1,c_z2),(w_z1,w_z2)) for every "row"" representing the GM parameterrs for 1 neuron beforehand
        This means that z_list ia s 3-dimensional-array with first = number of before neurons, second= mu,c,w, third = number of components
        The last entry in z_list is the bias neuron, which is deterministic and faked so it is like (1,0),(0,0)(0,0) but only the first one is important


        BUT: The last entry in z_list is deternministic as it comes from the bias neuron in the befrorehand layer

        returns a vector of the first ten moments of the pre activation value
    """

    #MISLEADING: BOTH INPUTS ARE ARRAYS
    num_moments = 10

    assert len(w_list) == len(z_list), "w_list and z_list must have the same length. but have lengths {} and {}".format(len(w_list), len(z_list))

    # Initialize result vector 
    result = np.zeros(num_moments, dtype=float)

    # Iterate through the list
    for i in range(len(w_list)-1):
        mu_w, c_w = w_list[i]
        mu_z, c_z, w_z = z_list[i]

        # Call the moments_pre_act_single function for each entry
        result += moments_pre_act_single(mu_w,c_w,mu_z,c_z,w_z)
    
    #Bias handling
    mu_w, c_w = w_list[-1]
    z_bias = z_list[-1,0,0]
    result += moments_pre_act_single_det(z_bias,mu_w,c_w)

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

def residuals_matching(params, t):
    """Compute the residual value for the optimization process."""
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

    # This should return a scalar value
    return residuals

def residual_sum_matching(params, t):
    """Compute the residual value for the optimization process."""
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
    
    # Compute the sum of the residuals
    residual_sum = np.sum(residuals)

    # This should return a scalar value
    return residual_sum


def match_moments(in_mom,components):
    """Function to perform the moment matching for given Moments and a desired number of components."""

    assert components < 3, "Number of components too hihg for number of moments"

    # Assemble the parameter vector according to the number of components. Per Component, we have 3 parameters: w, mu, c
    # The complete parameter vector will look like this: [w0,w1...,wN,mu0,mu1,...,muN,c0,c1,...,cN]
    params = np.zeros(components*3, dtype=float)

    #Set the initial guess for the weights as equal and summing up to one
    for i in range(components):
        params[i] = 1/components

    #Set the initial guess for the means as the input moments
    params[components] = 0.0    # Frst Mean is zero
    for i in range(components-1):
        params[components+i+1] = np.random.uniform(0,1)     # Rest of the means are random

    #Set the initial guess for the variances as the input moments
    for i in range(components):
        params[components*2+i] = 1.0     # Rest of the variances are rando

    print("Initial guess for the parameters:")
    print(params)

    # Optimize this with the minimize function to add in the constraint on the sum of the weights
    result = minimize(
        residuals_matching,
        params,
        args=(in_mom,),
        method="SLSQP",
        bounds=[(0, 1)] * (components * 3),
        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x[:components]) - 1}]
    )
    
    return result.x

def match_moments_2(in_mom, components):
    """Intermediate solution, only for two components"""
    if components != 2:
        raise ValueError("This function only works for two components")
    
    params = [0.5,0.0,1.0,1.0,1.0]  # Initial guess for [w0, mu0, mu1, c0, c1]

    # Call optimizer with bounds
    result = least_squares(residuals_weighted, params, args=in_mom, bounds=([0, -np.inf, -np.inf, 0, 0], [1.0, np.inf, np.inf, np.inf, np.inf]))

    w_res = [result.x[0], 1 - result.x[0]]
    mu_res = [result.x[1], result.x[2]]
    c_res = [result.x[3], result.x[4]]

    return mu_res, c_res, w_res


# Function to compute the residuals for optimization
def residuals(params, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
    w0_v, mu0_v, mu1_v, c0_v, c1_v = params
    w1_v = 1 - w0_v    #Attention! For more components we need to enforce the weights will be 1 in sum differently
    r = np.array([
        e1_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t1,
        e2_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t2,
        e3_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t3,
        e4_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t4,
        e5_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t5,
        e6_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t6,
        e7_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t7,
        e8_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t8,
        e9_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t9,
        e10_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t10
    ], dtype=float)

    return r

# Function to compute the residuals for optimization
def residuals_weighted(params, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
    w0_v, mu0_v, mu1_v, c0_v, c1_v = params
    w1_v = 1 - w0_v    #Attention! For more components we need to enforce the weights will be 1 in sum differently
    r = np.array([
        abs((e1_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t1) / t1),
        abs((e2_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t2) / t2),
        abs((e3_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t3) / t3),
        abs((e4_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t4) / t4),
        abs((e5_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t5) / t5),
        abs((e6_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t6) / t6),
        abs((e7_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t7) / t7),
        abs((e8_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t8) / t8),
        abs((e9_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t9) / t9),
        abs((e10_gm2(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t10) / t10)
    ], dtype=float)

    return r

# #******Store some stuff for two components and max order of 10******
# # Define the symbolic variables (exclude erf and erfc here)
# # a is the slope of the leaky part or the ReLU, c is the variance of the Gaussians, mu are the means of the GLM, w are the weights of the GLM
# a, c0, c1, mu0, mu1, w0 ,w1 = var('a c0 c1 mu0 mu1 w0 w1')

# # Gaussian Mixture Moments through leaky relu.Expression from mathematica (\[]-Style expressions need to be replaced without brackets)

# number_of_moments = 10

# #These are the formulas for 2 components from mathematica
# # ATTENTION: C IST STDDEVIATION, NOT VARIANCE!
# m_raw = [None] * (number_of_moments + 1) #We need to add the zeroth moment for the loop
# m_raw[0] = None #I dont use the zeroth moment!
# m_raw[1] = r"(c[0]/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a*c[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*mu[0])/2 + (a*Erfc[mu[0]/(Sqrt[2]*c[0])]*mu[0])/2)*w[0] + (c[1]/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a*c[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*mu[1])/2 + (a*Erfc[mu[1]/(Sqrt[2]*c[1])]*mu[1])/2)*w[1]"
# m_raw[2] = r"((c[0]*mu[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a^2*c[0]*mu[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*(c[0]^2 + mu[0]^2))/2 + (a^2*Erfc[mu[0]/(Sqrt[2]*c[0])]*(c[0]^2 + mu[0]^2))/2)*w[0] + ((c[1]*mu[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a^2*c[1]*mu[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*(c[1]^2 + mu[1]^2))/2 + (a^2*Erfc[mu[1]/(Sqrt[2]*c[1])]*(c[1]^2 + mu[1]^2))/2)*w[1]"
# m_raw[3] = r"((c[0]*(2*c[0]^2 + mu[0]^2))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a^3*c[0]*(2*c[0]^2 + mu[0]^2))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*mu[0]*(3*c[0]^2 + mu[0]^2))/2 + (a^3*Erfc[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(3*c[0]^2 + mu[0]^2))/2)*w[0] + ((c[1]*(2*c[1]^2 + mu[1]^2))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a^3*c[1]*(2*c[1]^2 + mu[1]^2))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*mu[1]*(3*c[1]^2 + mu[1]^2))/2 + (a^3*Erfc[mu[1]/(Sqrt[2]*c[1])]*mu[1]*(3*c[1]^2 + mu[1]^2))/2)*w[1]"
# m_raw[4] = r"((c[0]*mu[0]*(5*c[0]^2 + mu[0]^2))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a^4*c[0]*mu[0]*(5*c[0]^2 + mu[0]^2))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*(3*c[0]^4 + 6*c[0]^2*mu[0]^2 + mu[0]^4))/2 + (a^4*Erfc[mu[0]/(Sqrt[2]*c[0])]*(3*c[0]^4 + 6*c[0]^2*mu[0]^2 + mu[0]^4))/2)*w[0] + ((c[1]*mu[1]*(5*c[1]^2 + mu[1]^2))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a^4*c[1]*mu[1]*(5*c[1]^2 + mu[1]^2))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*(3*c[1]^4 + 6*c[1]^2*mu[1]^2 + mu[1]^4))/2 + (a^4*Erfc[mu[1]/(Sqrt[2]*c[1])]*(3*c[1]^4 + 6*c[1]^2*mu[1]^2 + mu[1]^4))/2)*w[1]"
# m_raw[5] = r"((c[0]*(c[0]^2 + mu[0]^2)*(8*c[0]^2 + mu[0]^2))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a^5*c[0]*(c[0]^2 + mu[0]^2)*(8*c[0]^2 + mu[0]^2))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*mu[0]*(15*c[0]^4 + 10*c[0]^2*mu[0]^2 + mu[0]^4))/2 + (a^5*Erfc[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(15*c[0]^4 + 10*c[0]^2*mu[0]^2 + mu[0]^4))/2)*w[0] + ((c[1]*(c[1]^2 + mu[1]^2)*(8*c[1]^2 + mu[1]^2))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a^5*c[1]*(c[1]^2 + mu[1]^2)*(8*c[1]^2 + mu[1]^2))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*mu[1]*(15*c[1]^4 + 10*c[1]^2*mu[1]^2 + mu[1]^4))/2 + (a^5*Erfc[mu[1]/(Sqrt[2]*c[1])]*mu[1]*(15*c[1]^4 + 10*c[1]^2*mu[1]^2 + mu[1]^4))/2)*w[1]"
# m_raw[6] = r"((c[0]*mu[0]*(33*c[0]^4 + 14*c[0]^2*mu[0]^2 + mu[0]^4))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a^6*c[0]*mu[0]*(33*c[0]^4 + 14*c[0]^2*mu[0]^2 + mu[0]^4))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*(15*c[0]^6 + 45*c[0]^4*mu[0]^2 + 15*c[0]^2*mu[0]^4 + mu[0]^6))/2 + (a^6*Erfc[mu[0]/(Sqrt[2]*c[0])]*(15*c[0]^6 + 45*c[0]^4*mu[0]^2 + 15*c[0]^2*mu[0]^4 + mu[0]^6))/2)*w[0] + ((c[1]*mu[1]*(33*c[1]^4 + 14*c[1]^2*mu[1]^2 + mu[1]^4))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a^6*c[1]*mu[1]*(33*c[1]^4 + 14*c[1]^2*mu[1]^2 + mu[1]^4))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*(15*c[1]^6 + 45*c[1]^4*mu[1]^2 + 15*c[1]^2*mu[1]^4 + mu[1]^6))/2 + (a^6*Erfc[mu[1]/(Sqrt[2]*c[1])]*(15*c[1]^6 + 45*c[1]^4*mu[1]^2 + 15*c[1]^2*mu[1]^4 + mu[1]^6))/2)*w[1]"
# m_raw[7] = r"((c[0]*(48*c[0]^6 + 87*c[0]^4*mu[0]^2 + 20*c[0]^2*mu[0]^4 + mu[0]^6))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a^7*c[0]*(48*c[0]^6 + 87*c[0]^4*mu[0]^2 + 20*c[0]^2*mu[0]^4 + mu[0]^6))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*mu[0]*(105*c[0]^6 + 105*c[0]^4*mu[0]^2 + 21*c[0]^2*mu[0]^4 + mu[0]^6))/2 + (a^7*Erfc[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(105*c[0]^6 + 105*c[0]^4*mu[0]^2 + 21*c[0]^2*mu[0]^4 + mu[0]^6))/2)*w[0] + ((c[1]*(48*c[1]^6 + 87*c[1]^4*mu[1]^2 + 20*c[1]^2*mu[1]^4 + mu[1]^6))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a^7*c[1]*(48*c[1]^6 + 87*c[1]^4*mu[1]^2 + 20*c[1]^2*mu[1]^4 + mu[1]^6))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*mu[1]*(105*c[1]^6 + 105*c[1]^4*mu[1]^2 + 21*c[1]^2*mu[1]^4 + mu[1]^6))/2 + (a^7*Erfc[mu[1]/(Sqrt[2]*c[1])]*mu[1]*(105*c[1]^6 + 105*c[1]^4*mu[1]^2 + 21*c[1]^2*mu[1]^4 + mu[1]^6))/2)*w[1]"
# m_raw[8] = r"((c[0]*mu[0]*(279*c[0]^6 + 185*c[0]^4*mu[0]^2 + 27*c[0]^2*mu[0]^4 + mu[0]^6))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) - (a^8*c[0]*mu[0]*(279*c[0]^6 + 185*c[0]^4*mu[0]^2 + 27*c[0]^2*mu[0]^4 + mu[0]^6))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*(105*c[0]^8 + 420*c[0]^6*mu[0]^2 + 210*c[0]^4*mu[0]^4 + 28*c[0]^2*mu[0]^6 + mu[0]^8))/2 + (a^8*Erfc[mu[0]/(Sqrt[2]*c[0])]*(105*c[0]^8 + 420*c[0]^6*mu[0]^2 + 210*c[0]^4*mu[0]^4 + 28*c[0]^2*mu[0]^6 + mu[0]^8))/2)*w[0] + ((c[1]*mu[1]*(279*c[1]^6 + 185*c[1]^4*mu[1]^2 + 27*c[1]^2*mu[1]^4 + mu[1]^6))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) - (a^8*c[1]*mu[1]*(279*c[1]^6 + 185*c[1]^4*mu[1]^2 + 27*c[1]^2*mu[1]^4 + mu[1]^6))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*(105*c[1]^8 + 420*c[1]^6*mu[1]^2 + 210*c[1]^4*mu[1]^4 + 28*c[1]^2*mu[1]^6 + mu[1]^8))/2 + (a^8*Erfc[mu[1]/(Sqrt[2]*c[1])]*(105*c[1]^8 + 420*c[1]^6*mu[1]^2 + 210*c[1]^4*mu[1]^4 + 28*c[1]^2*mu[1]^6 + mu[1]^8))/2)*w[1]"
# m_raw[9] = r"(-((a^9*c[0]*(384*c[0]^8 + 975*c[0]^6*mu[0]^2 + 345*c[0]^4*mu[0]^4 + 35*c[0]^2*mu[0]^6 + mu[0]^8))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (a^9*Erfc[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(945*c[0]^8 + 1260*c[0]^6*mu[0]^2 + 378*c[0]^4*mu[0]^4 + 36*c[0]^2*mu[0]^6 + mu[0]^8))/2 + (945*c[0]^8*mu[0] + 1260*c[0]^6*mu[0]^3 + 378*c[0]^4*mu[0]^5 + 36*c[0]^2*mu[0]^7 + mu[0]^9 + (Sqrt[2/Pi]*c[0]*(384*c[0]^8 + 975*c[0]^6*mu[0]^2 + 345*c[0]^4*mu[0]^4 + 35*c[0]^2*mu[0]^6 + mu[0]^8))/E^(mu[0]^2/(2*c[0]^2)) + Erf[mu[0]/(Sqrt[2]*c[0])]*(945*c[0]^8*mu[0] + 1260*c[0]^6*mu[0]^3 + 378*c[0]^4*mu[0]^5 + 36*c[0]^2*mu[0]^7 + mu[0]^9))/2)*w[0] + (-((a^9*c[1]*(384*c[1]^8 + 975*c[1]^6*mu[1]^2 + 345*c[1]^4*mu[1]^4 + 35*c[1]^2*mu[1]^6 + mu[1]^8))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi])) + (a^9*Erfc[mu[1]/(Sqrt[2]*c[1])]*mu[1]*(945*c[1]^8 + 1260*c[1]^6*mu[1]^2 + 378*c[1]^4*mu[1]^4 + 36*c[1]^2*mu[1]^6 + mu[1]^8))/2 + (945*c[1]^8*mu[1] + 1260*c[1]^6*mu[1]^3 + 378*c[1]^4*mu[1]^5 + 36*c[1]^2*mu[1]^7 + mu[1]^9 + (Sqrt[2/Pi]*c[1]*(384*c[1]^8 + 975*c[1]^6*mu[1]^2 + 345*c[1]^4*mu[1]^4 + 35*c[1]^2*mu[1]^6 + mu[1]^8))/E^(mu[1]^2/(2*c[1]^2)) + Erf[mu[1]/(Sqrt[2]*c[1])]*(945*c[1]^8*mu[1] + 1260*c[1]^6*mu[1]^3 + 378*c[1]^4*mu[1]^5 + 36*c[1]^2*mu[1]^7 + mu[1]^9))/2)*w[1]"
# m_raw[10]= r"((c[0]*mu[0]*(2895*c[0]^8 + 2640*c[0]^6*mu[0]^2 + 588*c[0]^4*mu[0]^4 + 44*c[0]^2*mu[0]^6 + mu[0]^8))/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[0]/(Sqrt[2]*c[0])])*(945*c[0]^10 + 4725*c[0]^8*mu[0]^2 + 3150*c[0]^6*mu[0]^4 + 630*c[0]^4*mu[0]^6 + 45*c[0]^2*mu[0]^8 + mu[0]^10))/2 + (a^10*(-((Sqrt[2/Pi]*c[0]*mu[0]*(2895*c[0]^8 + 2640*c[0]^6*mu[0]^2 + 588*c[0]^4*mu[0]^4 + 44*c[0]^2*mu[0]^6 + mu[0]^8))/E^(mu[0]^2/(2*c[0]^2))) + Erfc[mu[0]/(Sqrt[2]*c[0])]*(945*c[0]^10 + 4725*c[0]^8*mu[0]^2 + 3150*c[0]^6*mu[0]^4 + 630*c[0]^4*mu[0]^6 + 45*c[0]^2*mu[0]^8 + mu[0]^10)))/2)*w[0] + ((c[1]*mu[1]*(2895*c[1]^8 + 2640*c[1]^6*mu[1]^2 + 588*c[1]^4*mu[1]^4 + 44*c[1]^2*mu[1]^6 + mu[1]^8))/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + Erf[mu[1]/(Sqrt[2]*c[1])])*(945*c[1]^10 + 4725*c[1]^8*mu[1]^2 + 3150*c[1]^6*mu[1]^4 + 630*c[1]^4*mu[1]^6 + 45*c[1]^2*mu[1]^8 + mu[1]^10))/2 + (a^10*(-((Sqrt[2/Pi]*c[1]*mu[1]*(2895*c[1]^8 + 2640*c[1]^6*mu[1]^2 + 588*c[1]^4*mu[1]^4 + 44*c[1]^2*mu[1]^6 + mu[1]^8))/E^(mu[1]^2/(2*c[1]^2))) + Erfc[mu[1]/(Sqrt[2]*c[1])]*(945*c[1]^10 + 4725*c[1]^8*mu[1]^2 + 3150*c[1]^6*mu[1]^4 + 630*c[1]^4*mu[1]^6 + 45*c[1]^2*mu[1]^8 + mu[1]^10)))/2)*w[1]"

# # Turn Mathematica expressions into usable python expressions
# m = [None] *(number_of_moments)

# for i,m in enumerate(m_raw):
#     if i == 0:
#         continue
#     # Replace \[Pi] with Pi in the expressions so python can read it
#     m_raw[i] = replace_pi(m_raw[i])
#     # Replace mu[i] with mui when i is a variable integer
#     m_raw[i] = replace_mu_index(m_raw[i])
#     # Replace w[i] with wi when i is a variable integer
#     m_raw[i] = replace_w_index(m_raw[i])
#     # Replace c[i] with ci when i is a variable integer
#     m_raw[i] = replace_c_index(m_raw[i])

# #Parse every expression for translation
# mp = []
# for mx in m_raw:
#     if mx is None:
#         continue
#     mp.append(parse_mathematica(mx)) 

# #Replace the function expressions with handles
# for i, mx in enumerate(mp):
#     mp[i] = mx.replace(
#         lambda x: x.func.__name__ == 'Erf',
#         lambda x: erf(x.args[0])
#     ).replace(
#         lambda x: x.func.__name__ == 'Erfc',
#         lambda x: erfc(x.args[0])
#     )   

# print("Parsed Mathematica expressions:")
# for i, mx in enumerate(mp):
#     print(f"m[{i}] = {mx}")
# print("")

# def compute_moments_analytic(a_test, c0_test, c1_test, mu0_test, mu1_test, w0_test, w1_test):
#     """
#     Compute the moments for a two component Gaussian Mixture analytically using the provided parameters.
#     c is the covariance, will be replaced with stddev in the function itself
#     """
#     # Substitute Values (Take care for variance and stddev)
#     values = {a: a_test, c0: np.sqrt(c0_test), c1:np.sqrt(c1_test), mu0: mu0_test, mu1:mu1_test, w0: w0_test, w1: w1_test}
#     moments_analytic = []
#     for i, mx in enumerate(mp):
#         evaluated_expr = mx.subs(values)
#         # Numerically evaluate the result
#         result = evaluated_expr.evalf()
#         moments_analytic.append(result)
#     return moments_analytic

def fit_gm_moments(params,args):
    """
    Fit the Gaussian Mixture model to the moments using least squares optimization.
    """
    # Call optimizer with bounds
    result = least_squares(residuals_weighted, params, args=args, bounds=([0, -np.inf, -np.inf, 0, 0], [1.0, np.inf, np.inf, np.inf, np.inf]))
    return result


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
            weight_mean = np.random.normal(0, 1, (self.layers[i]+1, self.layers[i+1]))+1
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

    def forward_moments(self,x):
        """
        Forward pass through the network based on the method of moments.
        x: Input data, deterministic
        """
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
            w_list = np.stack((self.weight_means[0][:,i], self.weight_variances[0][:,i]), axis=1)
            moments_pre = moments_pre_act_combined_det(x,w_list)

            # Match the GM parameters to the moments
            means, variances, weights = match_moments_2(moments_pre, components = 2)

            self.means_gm_pre[0][i,:] = np.array(means)
            self.variances_gm_pre[0][i,:] = np.array(variances)
            self.weights_gm_pre[0][i,:] = np.array(weights)

            # Match Post-Activation
            # Compute moments of the post activation
            print(f"Call PA in Neuron {i}, layer 0")
            moments_post =  moments_post_act(self.a_relu, means, variances, weights)

            # Match the GM parameters to the moments
            means, variances, weights = match_moments_2(moments_post, components = 2)
            self.means_gm_post[0][i,:] = np.array(means)
            self.variances_gm_post[0][i,:] = np.array(variances)
            self.weights_gm_post[0][i,:] = np.array(weights)

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
                z_complete = np.concatenate((z_complete, np.zeros((1, 3, 2))), axis=0)
                z_complete[-1, 0, 0] = 1 # Bias as fictive GM component with mean 1
                w_complete = np.stack((self.weight_means[l][:,i], self.weight_variances[l][:,i]), axis=1)
                moments_pre = moments_pre_act_combined(w_complete, z_complete)

                # Match the GM parameters to the moments
                means, variances, weights = match_moments_2(moments_pre, components = 2)

                self.means_gm_pre[l][i,:] = np.array(means)
                self.variances_gm_pre[l][i,:] = np.array(variances)
                self.weights_gm_pre[l][i,:] = np.array(weights)

                # Match Post-Activation
                if self.activations[l-1] == 'relu':
                    # Compute moments of the post activation
                    print(f"Call PA in Neuron {i}, layer {l}..")
                    moments_post =  moments_post_act(self.a_relu, means, variances, weights)

                    # Match the GM parameters to the moments
                    means, variances, weights = match_moments_2(moments_post, components = 2)
                    
                elif self.activations[l-1] == 'linear':
                    moments_post = moments_pre

                self.means_gm_post[l][i,:] = np.array(means)
                self.variances_gm_post[l][i,:] = np.array(variances)
                self.weights_gm_post[l][i,:] = np.array(weights)

        ######
        return self.means_gm_post[-1], self.variances_gm_post[-1], self.weights_gm_post[-1]
