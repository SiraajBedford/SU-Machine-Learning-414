'''
Computes responsibilities. Assumes one-dimensional data and a two component mixture model.

@param p: mixture coefficients.
@type p: 1-dimensional list of floats of length 2.
@param u: class means.
@type u: 1-dimensional list of floats length 2.
@param s: class standard deviations.
@type s: 1-dimensional list of floats of length 2. 
@param x: vector of scalar observations
@type x: 1-dimensional list of floats of length n.
@param c: class label
@type c: 1 or 0 [integer]

@return: the calculated responsibility of each observation associated with class c
@rtype: 1-dimensional list of floats of length n
'''
def estimate_gamma(p,u,s,x,c):
    import math
    from math import exp
    from math import sqrt
    #p[0] mixture coefficient for class 0
    #p[1] mixture coefficient for class 1
    
    #u[0] mean of class 0
    #u[1] mean of class 1
    
    #s[0] std of class 0
    #s[1] std of class 1
     
    #c class label (can be either a 0 or 1) 
    
    #x vector of scalar observations
    
    #You may assume that x will not be empty and that the user will provide valid inputs.
    
    g = [None]*len(x) #responsibilities
    
    for k in range(len(x)):
        t0 = (1.0/(sqrt(2*math.pi*s[0]**2)))*exp(-1*((x[k] - u[0])**2)/(2*s[0]**2)) 
        t1 = (1.0/(sqrt(2*math.pi*s[1]**2)))*exp(-1*((x[k] - u[1])**2)/(2*s[1]**2)) 
        if c == 0:
            g[k] = (p[0]*t0)/(p[0]*t0+p[1]*t1)
        else:
            g[k] = (p[1]*t1)/(p[1]*t1+p[0]*t0)
    
    return g