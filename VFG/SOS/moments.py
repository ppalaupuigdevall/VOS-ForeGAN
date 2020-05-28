""" 
Author: Marina Alonso

The notation from "SoS-RSC: A Sum-of-Squares Polynomial Approach to Robustifying Subspace Clustering Algorithms", section 2.

x = [x1, x2, ... xd]

          n+d  
s_nd = (       )
           d

Mn: Moment matrix (s_nd, s_nd)

v_n(x): veronese map of x: all possible monomials of order n in d variables in lexicographical order

"""

import numpy as np
import torch
from scipy.special import comb
import matplotlib.pyplot as plt
import scipy.stats as ss

def generateMoments(hist, ord, d):
    """
    """
    # d is the dimension of our data, d is 1 for a scalar
    s_nd = int(comb(ord//2 + d , d))
    z = np.linspace(0.0,1.0,len(hist))
    a = np.zeros(ord+1)
    for i in range(0,ord +1):
        a[i] = np.sum((z**i)*hist)
    M = np.zeros((s_nd, s_nd))
    for i in range(0, s_nd):
        for j in range(0, s_nd):
            M[i,j] = a[i+j] 
    return M


def Q(M, z):
    z = z.reshape(len(z),1)
    M_inv = np.linalg.inv(M)
    veronese = np.zeros((len(z), M.shape[0]))
    for i in range(0, M.shape[0]):
        veronese[:,i] = (z**i).reshape(len(z))
    veronese_T = veronese.T
    q_eval = np.matmul(veronese,np.matmul(M_inv, veronese_T))
    # This was wrong, we just have to keep the i,i value of q_eval
    q_final = q_eval.diagonal()
    return q_final


if __name__ == "__main__":
    print('Main')
    # Code is this main section is intended to test the functions defined above
    
    x = np.random.normal(0.5,0.1,20000)
    
    hist, x_axis, _ = plt.hist(x, bins = 200)
    
    x_axis = x_axis[:-1]
    hist = hist/np.sum(hist)
    ord_g = 4
    M = generateMoments(hist, ord_g,1)
    magic_q = comb(1+ord_g, 1)
    print(magic_q)
    q_eval = Q(M, x_axis)
    
    plt.subplot(211)
    plt.title("Gaussian Distr. mu=0.5, ss=0.1")
    plt.plot(x_axis, hist)
    plt.subplot(212)
    plt.title("Q(x) with M"+str(ord_g))
    plt.plot(x_axis, q_eval)
    plt.plot(x_axis, magic_q*np.ones(len(x_axis)))
    plt.show()


    # MIXTURE OF GAUSSIANS
    # Set-up.
    n = 20000
    np.random.seed(0x5eed)
    # Parameters of the mixture components
    norm_params = np.array([[0, 0.1],
                            [1, 0.2]])
                            # [9, 1.3]])
    n_components = norm_params.shape[0]
    # Weight of each component, in this case all of them are 1/3
    weights = np.ones(n_components, dtype=np.float64) / 2.0
    
    # A stream of indices from which to choose the component
    mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
    # y is the mixture sample
    x = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                    dtype=np.float64)
    
    hist, x_axis, _ = plt.hist(x, bins = 2000)

    x_axis = x_axis[:-1]

    hist = hist/np.sum(hist)
    ord_m = 12
    M = generateMoments(hist,ord_m,1)

    q_eval = Q(M, x_axis)

    plt.subplot(211)
    plt.title("mixture of gaussians")
    plt.plot(x_axis, hist)
    plt.subplot(212)
    plt.title("log(Q(x)) with M"+str(ord_m))
    plt.plot(x_axis, np.log(q_eval))
    plt.plot(x_axis, np.log(magic_q)*np.ones(len(x_axis)))
    # plt.plot(x_axis, q_eval)

    plt.show()