import numpy as np
import torch
from scipy.special import comb


def exponent_nk(n, K):
    id = np.diag(np.ones(K))
    exp = id

    for i in range(1, n):
        rene = np.asarray([])
        for j in range(0, K):
            for k in range(exp.shape[0] - int(comb(i+K-j-1, i)), exp.shape[0]):
                if rene.shape[0] == 0:
                    rene = id[j, :]+exp[k, :]
                    rene = np.expand_dims(rene, axis=0)
                else:
                    rene = np.concatenate([rene, np.expand_dims(id[j, :]+exp[k, :], axis=0)], axis=0)
        exp = rene.copy()
    return exp


def veronese_nk(x, n, if_cuda=False, if_coffiecnt=False):
    '''
     Computes the Veronese map of degree n, that is all
     the monomials of a certain degree.
     x is a K by N matrix, where K is dimension and N number of points
     y is a K by Mn matrix, where Mn = nchoosek(n+K-1,n)
     powes is a K by Mn matrix with the exponent of each monomial

     Example veronese([x1;x2],2) gives
     y = [x1^2;x1*x2;x2^2]
     powers = [2 0; 1 1; 0 2]

     Copyright @ Rene Vidal, 2003
    '''

    if if_coffiecnt:
        assert n == 2
    K, N = x.shape[0], x.shape[1]
    powers = exponent_nk(n, K)
    if if_cuda:
        powers = torch.tensor(powers, dtype=torch.float).cuda('cuda:2')
        
    else:
        powers = torch.tensor(powers, dtype=torch.float)
    if n == 0:
        y = 1
    elif n == 1:
        y = x
    else:
        # x[x <= 1e-10] = 1e-10
        # y = np.real(np.exp(np.matmul(powers, np.log(x))))
        s = []
        for i in range(0, powers.shape[0]):
            # if powers[i, :].sum() == 0:
            #     s.append(torch.ones([1, x.shape[1]]))
            # else:
            #     tmp = x.t().pow(powers[i, :])
            #     ind = torch.ge(powers[i, :].expand_as(tmp), 1).float()
            #     s.append(torch.mul(tmp, ind).sum(dim=1).unsqueeze(dim=0))
            tmp = x.t().pow(powers[i, :])
            ttmp = tmp[:, 0]
            for j in range(1, tmp.shape[1]):
                ttmp = torch.mul(ttmp, tmp[:, j])
            if if_coffiecnt:
                if powers[i, :].max() == 1:
                    ttmp = ttmp * 1.4142135623730951
            s.append(ttmp.unsqueeze(dim=0))
        y = torch.cat(s, dim=0)
    return y, powers


def generate_veronese(x, n):
    """Concatenates the results of veronese_nk function to generate the complete veronese map of x
        @param x: Matrix (dim, npoints)
        @param n: the veronese map will be up to degree n
        
        Output: the complete veronese map of x (veronese_dim, BS)
        Example:
        if x is a two dimensional vector x = [x1 x2]
        generate_veronese(x, 2) ==> [1 x1 x2 x1^2 x1*x2 x2^2]
        generate_veronese(x, 3) ==> [1 x1 x2 x1^2 x1*x2 x2^2 x1^3 x1^2*x2 x1*x2^2 x2^3]
    """
    v_x = x
    
    p_x = None
    for i in range(0,n-1):
        v_x_n, p_n = veronese_nk(x, i+2,if_cuda=True,if_coffiecnt=False,)
        v_x = torch.cat([v_x, v_x_n], dim=0)
    
    v_x = torch.cat([torch.ones(1,v_x.size()[1]).cuda('cuda:2'), v_x])
    return v_x, p_x


if __name__ == "__main__":
    # d = 64
    # BS = 32
    x = torch.tensor([1.0,2.0,3.0])
    # x = torch.rand([d,BS])
    x1 = torch.cat([torch.ones([1, BS]), x])
    n = 2 # degree of the polynomial, this will generate a moment matrix up to 2*n
    y, p = veronese_nk(x, n, if_cuda=False, if_coffiecnt=False)

    d = 64
    BS = 100
    # x = torch.tensor([1.0,2.0,3.0])
    x = torch.rand([d,BS])+1.0
    # x1 = torch.cat([torch.ones([1, BS]), x])
    n = 2 # degree of the polynomial, this will generate a moment matrix up to 2*n
    # y, p = veronese_nk(x, n, if_cuda=False, if_coffiecnt=False)
    y, p = generate_veronese(x,n)
    print("size y " + str(y.size()))
    print(y)
    dim_v = int(comb(d+n, n))
    print("dim_v : " + str(dim_v))
    V = torch.matmul(y.view(BS, dim_v ,1), y.view(BS, 1,dim_v))
    V = torch.mean(V,dim=0)
    print(V)
    print(y)
    M_inv = torch.inverse(V)
    print(M_inv)
    v_x_test, _ = generate_veronese(torch.rand([64,1]),n)
    Q = torch.matmul(torch.matmul(v_x_test.view(1,1,dim_v),M_inv), v_x_test.view(1,dim_v,1))
    print(Q)
    v_x_test, _ = generate_veronese(torch.rand([64,1])*20.0,n)
    Q = torch.matmul(torch.matmul(v_x_test.view(1,1,dim_v),M_inv), v_x_test.view(1,dim_v,1))
    print(Q)