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
from scipy.optimize import least_squares

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
def e6(mu, c_v):
    return 15*c_v**3+45*(c_v**2)*(mu**2)+15*mu**4+mu**6
def e7(mu, c_v):
    return 105*mu*(c_v**3)+105*(c_v**2)*(mu**3)+21*(mu**5)*c_v+mu**7
def e8(mu, c_v):    
    return 105*c_v**4+420*(c_v**3)*(mu**2)+210*(c_v**2)*(mu**4)+28*(c_v)*(mu**6)+mu**8
def e9(mu, c_v):
    return 945*(c_v**4)*mu+1260*(c_v**3)*(mu**3)+378*(c_v**2)*(mu**5)+36*c_v*(mu**7)+mu**9
def e10(mu, c_v):
    return 945*c_v**5+4725*(c_v**4)*(mu**2)+3150*(c_v**3)*(mu**4)+630*(c_v**2)*(mu**6)+45*(c_v)*mu**8+mu**10

# Noncentral moments of a Gaussian Mixture
def e1_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e1(mu0_v) + w1_v*e1(mu1_v))

def e2_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e2(mu0_v,c0_v) + w1_v*e2(mu1_v,c1_v))

def e3_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e3(mu0_v,c0_v) + w1_v*e3(mu1_v,c1_v))

def e4_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e4(mu0_v,c0_v) + w1_v*e4(mu1_v,c1_v))

def e5_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e5(mu0_v,c0_v) + w1_v*e5(mu1_v,c1_v))

def e6_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e6(mu0_v,c0_v) + w1_v*e6(mu1_v,c1_v))

def e7_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e7(mu0_v,c0_v) + w1_v*e7(mu1_v,c1_v))

def e8_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e8(mu0_v,c0_v) + w1_v*e8(mu1_v,c1_v))

def e9_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e9(mu0_v,c0_v) + w1_v*e9(mu1_v,c1_v))

def e10_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v):
    return (w0_v*e10(mu0_v,c0_v) + w1_v*e10(mu1_v,c1_v))

# Function to compute the residuals for optimization
def residuals(params, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
    w0_v, mu0_v, mu1_v, c0_v, c1_v = params
    w1_v = 1 - w0_v    #Attention! For more components we need to enforce the weights will be 1 in sum differently
    r = np.array([
        e1_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t1,
        e2_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t2,
        e3_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t3,
        e4_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t4,
        e5_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t5,
        e6_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t6,
        e7_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t7,
        e8_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t8,
        e9_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t9,
        e10_gm(w0_v,w1_v,mu0_v,mu1_v,c0_v,c1_v) - t10
    ], dtype=float)

    return r

#******Store some stuff for two components and max order of 10******
# Define the symbolic variables (exclude erf and erfc here)
# a is the slope of the leaky part or the ReLU, c is the variance of the Gaussians, mu are the means of the GLM, w are the weights of the GLM
a, c0, c1, mu0, mu1, w0 ,w1 = var('a c0 c1 mu0 mu1 w0 w1')

# Gaussian Mixture Moments through leaky relu.Expression from mathematica (\[]-Style expressions need to be replaced without brackets)

number_of_moments = 10

#These are the formulas for 2 components from mathematica
m_raw = [None] * (number_of_moments + 1) #We need to add the zeroth moment for the loop
m_raw[0] = None #I dont use the zeroth moment!
m_raw[1] = r"-(((-1 + a)*c[0]*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*mu[0]*w[0])/2 - ((-1 + a)*c[1]*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a*Erfc[mu[0]/(Sqrt[2]*c[0])])*mu[0]*w[0] + (2 + (-1 + a)*Erfc[mu[1]/(Sqrt[2]*c[1])])*mu[1]*w[1])/2"""
m_raw[2] = r"-(((-1 + a^2)*c[0]*mu[0]*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*(c[0]^2 + mu[0]^2)*w[0])/2 - ((-1 + a^2)*c[1]*mu[1]*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^2*Erfc[mu[0]/(Sqrt[2]*c[0])])*(c[0]^2 + mu[0]^2)*w[0] + (2 + (-1 + a^2)*Erfc[mu[1]/(Sqrt[2]*c[1])])*(c[1]^2 + mu[1]^2)*w[1])/2"
m_raw[3] = r"-(((-1 + a^3)*c[0]*(2*c[0]^2 + mu[0]^2)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(3*c[0]^2 + mu[0]^2)*w[0])/2 - ((-1 + a^3)*c[1]*(2*c[1]^2 + mu[1]^2)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^3*Erfc[mu[0]/(Sqrt[2]*c[0])])*mu[0]*(3*c[0]^2 + mu[0]^2)*w[0] + (2 + (-1 + a^3)*Erfc[mu[1]/(Sqrt[2]*c[1])])*mu[1]*(3*c[1]^2 + mu[1]^2)*w[1])/2"
m_raw[4] = r"-(((-1 + a^4)*c[0]*mu[0]*(5*c[0]^2 + mu[0]^2)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*(3*c[0]^4 + 6*c[0]^2*mu[0]^2 + mu[0]^4)*w[0])/2 - ((-1 + a^4)*c[1]*mu[1]*(5*c[1]^2 + mu[1]^2)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^4*Erfc[mu[0]/(Sqrt[2]*c[0])])*(3*c[0]^4 + 6*c[0]^2*mu[0]^2 + mu[0]^4)*w[0] + (2 + (-1 + a^4)*Erfc[mu[1]/(Sqrt[2]*c[1])])*(3*c[1]^4 + 6*c[1]^2*mu[1]^2 + mu[1]^4)*w[1])/2"
m_raw[5] = r"-(((-1 + a^5)*c[0]*(c[0]^2 + mu[0]^2)*(8*c[0]^2 + mu[0]^2)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(15*c[0]^4 + 10*c[0]^2*mu[0]^2 + mu[0]^4)*w[0])/2 - ((-1 + a^5)*c[1]*(c[1]^2 + mu[1]^2)*(8*c[1]^2 + mu[1]^2)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^5*Erfc[mu[0]/(Sqrt[2]*c[0])])*mu[0]*(15*c[0]^4 + 10*c[0]^2*mu[0]^2 + mu[0]^4)*w[0] + (2 + (-1 + a^5)*Erfc[mu[1]/(Sqrt[2]*c[1])])*mu[1]*(15*c[1]^4 + 10*c[1]^2*mu[1]^2 + mu[1]^4)*w[1])/2"
m_raw[6] = r"-(((-1 + a^6)*c[0]*mu[0]*(33*c[0]^4 + 14*c[0]^2*mu[0]^2 + mu[0]^4)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*(15*c[0]^6 + 45*c[0]^4*mu[0]^2 + 15*c[0]^2*mu[0]^4 + mu[0]^6)*w[0])/2 - ((-1 + a^6)*c[1]*mu[1]*(33*c[1]^4 + 14*c[1]^2*mu[1]^2 + mu[1]^4)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^6*Erfc[mu[0]/(Sqrt[2]*c[0])])*(15*c[0]^6 + 45*c[0]^4*mu[0]^2 + 15*c[0]^2*mu[0]^4 + mu[0]^6)*w[0] + (2 + (-1 + a^6)*Erfc[mu[1]/(Sqrt[2]*c[1])])*(15*c[1]^6 + 45*c[1]^4*mu[1]^2 + 15*c[1]^2*mu[1]^4 + mu[1]^6)*w[1])/2"
m_raw[7] = r"-(((-1 + a^7)*c[0]*(48*c[0]^6 + 87*c[0]^4*mu[0]^2 + 20*c[0]^2*mu[0]^4 + mu[0]^6)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(105*c[0]^6 + 105*c[0]^4*mu[0]^2 + 21*c[0]^2*mu[0]^4 + mu[0]^6)*w[0])/2 - ((-1 + a^7)*c[1]*(48*c[1]^6 + 87*c[1]^4*mu[1]^2 + 20*c[1]^2*mu[1]^4 + mu[1]^6)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^7*Erfc[mu[0]/(Sqrt[2]*c[0])])*mu[0]*(105*c[0]^6 + 105*c[0]^4*mu[0]^2 + 21*c[0]^2*mu[0]^4 + mu[0]^6)*w[0] + (2 + (-1 + a^7)*Erfc[mu[1]/(Sqrt[2]*c[1])])*mu[1]*(105*c[1]^6 + 105*c[1]^4*mu[1]^2 + 21*c[1]^2*mu[1]^4 + mu[1]^6)*w[1])/2"
m_raw[8] = r"-(((-1 + a^8)*c[0]*mu[0]*(279*c[0]^6 + 185*c[0]^4*mu[0]^2 + 27*c[0]^2*mu[0]^4 + mu[0]^6)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*(105*c[0]^8 + 420*c[0]^6*mu[0]^2 + 210*c[0]^4*mu[0]^4 + 28*c[0]^2*mu[0]^6 + mu[0]^8)*w[0])/2 - ((-1 + a^8)*c[1]*mu[1]*(279*c[1]^6 + 185*c[1]^4*mu[1]^2 + 27*c[1]^2*mu[1]^4 + mu[1]^6)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^8*Erfc[mu[0]/(Sqrt[2]*c[0])])*(105*c[0]^8 + 420*c[0]^6*mu[0]^2 + 210*c[0]^4*mu[0]^4 + 28*c[0]^2*mu[0]^6 + mu[0]^8)*w[0] + (2 + (-1 + a^8)*Erfc[mu[1]/(Sqrt[2]*c[1])])*(105*c[1]^8 + 420*c[1]^6*mu[1]^2 + 210*c[1]^4*mu[1]^4 + 28*c[1]^2*mu[1]^6 + mu[1]^8)*w[1])/2"
m_raw[9] = r"-(((-1 + a^9)*c[0]*(384*c[0]^8 + 975*c[0]^6*mu[0]^2 + 345*c[0]^4*mu[0]^4 + 35*c[0]^2*mu[0]^6 + mu[0]^8)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*mu[0]*(945*c[0]^8 + 1260*c[0]^6*mu[0]^2 + 378*c[0]^4*mu[0]^4 + 36*c[0]^2*mu[0]^6 + mu[0]^8)*w[0])/2 - ((-1 + a^9)*c[1]*(384*c[1]^8 + 975*c[1]^6*mu[1]^2 + 345*c[1]^4*mu[1]^4 + 35*c[1]^2*mu[1]^6 + mu[1]^8)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^9*Erfc[mu[0]/(Sqrt[2]*c[0])])*mu[0]*(945*c[0]^8 + 1260*c[0]^6*mu[0]^2 + 378*c[0]^4*mu[0]^4 + 36*c[0]^2*mu[0]^6 + mu[0]^8)*w[0] + (2 + (-1 + a^9)*Erfc[mu[1]/(Sqrt[2]*c[1])])*mu[1]*(945*c[1]^8 + 1260*c[1]^6*mu[1]^2 + 378*c[1]^4*mu[1]^4 + 36*c[1]^2*mu[1]^6 + mu[1]^8)*w[1])/2"
m_raw[10] = r"-(((-1 + a^10)*c[0]*mu[0]*(2895*c[0]^8 + 2640*c[0]^6*mu[0]^2 + 588*c[0]^4*mu[0]^4 + 44*c[0]^2*mu[0]^6 + mu[0]^8)*w[0])/(E^(mu[0]^2/(2*c[0]^2))*Sqrt[2*Pi])) + (Erf[mu[0]/(Sqrt[2]*c[0])]*(945*c[0]^10 + 4725*c[0]^8*mu[0]^2 + 3150*c[0]^6*mu[0]^4 + 630*c[0]^4*mu[0]^6 + 45*c[0]^2*mu[0]^8 + mu[0]^10)*w[0])/2 - ((-1 + a^10)*c[1]*mu[1]*(2895*c[1]^8 + 2640*c[1]^6*mu[1]^2 + 588*c[1]^4*mu[1]^4 + 44*c[1]^2*mu[1]^6 + mu[1]^8)*w[1])/(E^(mu[1]^2/(2*c[1]^2))*Sqrt[2*Pi]) + ((1 + a^10*Erfc[mu[0]/(Sqrt[2]*c[0])])*(945*c[0]^10 + 4725*c[0]^8*mu[0]^2 + 3150*c[0]^6*mu[0]^4 + 630*c[0]^4*mu[0]^6 + 45*c[0]^2*mu[0]^8 + mu[0]^10)*w[0] + (2 + (-1 + a^10)*Erfc[mu[1]/(Sqrt[2]*c[1])])*(945*c[1]^10 + 4725*c[1]^8*mu[1]^2 + 3150*c[1]^6*mu[1]^4 + 630*c[1]^4*mu[1]^6 + 45*c[1]^2*mu[1]^8 + mu[1]^10)*w[1])/2"

# Turn Mathematica expressions into usable python expressions
m = [None] *(number_of_moments)

for i,m in enumerate(m_raw):
    if i == 0:
        continue
    # Replace \[Pi] with Pi in the expressions so python can read it
    m_raw[i] = replace_pi(m_raw[i])
    # Replace mu[i] with mui when i is a variable integer
    m_raw[i] = replace_mu_index(m_raw[i])
    # Replace w[i] with wi when i is a variable integer
    m_raw[i] = replace_w_index(m_raw[i])
    # Replace c[i] with ci when i is a variable integer
    m_raw[i] = replace_c_index(m_raw[i])

#Parse every expression for translation
mp = []
for mx in m_raw:
    if mx is None:
        continue
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

def compute_moments_analytic(a_test, c0_test, c1_test, mu0_test, mu1_test, w0_test, w1_test):
    """
    Compute the moments for a two component Gaussian Mixture analytically using the provided parameters.
    """
    # Substitute Values
    values = {a: a_test, c0: c0_test, c1:c1_test, mu0: mu0_test, mu1:mu1_test, w0: w0_test, w1: w1_test}
    moments_analytic = []
    for i, mx in enumerate(mp):
        evaluated_expr = mx.subs(values)
        # Numerically evaluate the result
        result = evaluated_expr.evalf()
        moments_analytic.append(result)
    return moments_analytic

def fit_gm_moments(params,args):
    """
    Fit the Gaussian Mixture model to the moments using least squares optimization.
    """
    # Call optimizer with bounds
    result = least_squares(residuals,params,args=args)
    return result
