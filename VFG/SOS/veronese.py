import numpy as np
import torch
# torch.set_default_dtype(torch.float64)
from scipy.special import comb
torch.cuda.set_device(2)
# torch.manual_seed(0)
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
        powers = torch.tensor(powers, dtype=torch.float32).cuda()
    else:
        powers = torch.tensor(powers, dtype=torch.float32)
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
        v_x_n, p_n = veronese_nk(x, i+2,if_cuda=True,if_coffiecnt=False)
        v_x = torch.cat([v_x, v_x_n], dim=0)

    v_x = torch.cat([torch.ones(1,v_x.size()[1]).cuda(), v_x])
    return v_x, p_x

import torch.nn as nn
class Q_real_M(nn.Module):
    """
    This module is in charge of 
        1. Building a moment matrix with the training samples (INLIERS)
            2. Applying the inverse of the empirically built moment matrix to discriminate outliers/inliers
    
    Basically, M = sum{v(x)*v.T(x)}
    Then applies M_inv:

                v(x).T * M_inv * v(x)
    """
    def __init__(self, x_size, n):
        """
        x_size = vector_size, [x1 x2 ... xd]
        n = moment degree up to n
        """
        super(Q_real_M, self).__init__()
        self.n = n
        self.dim_veronese = int(comb(x_size + n, n))
        self.veroneses = []
        self.has_M_inv = False 
        self.M_inv = None
        self.build_M = True # Will be true when the autoencoder gets good reconstruction

    def forward(self, x):
        if((not self.has_M_inv) and (self.build_M)):
            # We want only veronese maps to build the moment matrix once we have good reconstruction (self.build_M = True) !
            npoints, dims = x.size()
            v_x, _ = generate_veronese(x.view(dims, npoints).cuda(), self.n)
            self.veroneses.append(v_x.cpu())
        elif(self.has_M_inv):
            # Create the veronese map of z
            npoints, dims = x.size()
            # print("npoints = ",npoints, " dims = ",dims)
            v_x, _ = generate_veronese(x.view(dims, npoints).cuda(), self.n)
            
            dim_veronese, BS = v_x.size()
            # print("npoints = ",BS, " dims = ",dim_veronese)
            # first = torch.matmul(v_x.view(BS,1,dim_veronese), self.M_inv)
            # x = torch.matmul(first, v_x.view(BS, dim_veronese, 1))
            # x = torch.matmul(
            #     v_x.view(BS, 1, dim_veronese), torch.matmul(self.M_inv,
            #     v_x.view(BS, dim_veronese, 1)))
            
            # BATCH version, careful because it consumes a lot of gpu memory
            x = torch.matmul(
                v_x.view(BS, 1, dim_veronese), torch.matmul(torch.cat(BS*[self.M_inv], dim=0),
                v_x.view(BS, dim_veronese, 1)))
            
        return x

    def create_M(self):
        # This method should be created from outlise the class
        
        with torch.no_grad():
            n = len(self.veroneses)
            d, bs = self.veroneses[0].size()
            Mc = torch.tensor([])
            for i in range(0,n):
                print(i/n)
                #V = torch.cat([V, self.veroneses[i+1]], dim=1)
                V = self.veroneses[i]
                A = self.veroneses[i]
                A = A.view(bs,d,1)
                B = self.veroneses[i]
                B = B.view(bs,1,d)
                Mc_m = torch.matmul(A, B)
                # Mc_m = torch.bmm(V.view(bs,d,1), V.view(bs,1,d))
                # print(Mc_m)
                Mc_m = torch.mean(Mc_m, dim=0)
                Mc = torch.cat([Mc,Mc_m.unsqueeze(0)])
                # print(Mc.size())
            M = torch.mean(Mc,dim=0) + 0.0001 * torch.eye(d)
            # M = torch.mean(Mc,dim=0)
            # print(M.size())
            # print("Moment matrix, should have hankel structure")
            # print(M)
            # U_m, S_m, V_m = torch.svd(M)
            # print("singular values of M = ", S_m)
            self.M_inv = torch.inverse(M).cuda()
            print(self.M_inv)
            # U,S,V = torch.svd(self.M_inv)
            # print("Singular values of M_inv = ", S)
            self.has_M_inv = True
    
    def set_build_M(self):
        self.build_M = True
        self.create_M()

class Q_real_M_2(nn.Module):
    """
    This module is in charge of 
        1. Building a moment matrix with the training samples (INLIERS)
            2. Applying the inverse of the empirically built moment matrix to discriminate outliers/inliers
    
    Basically, M = sum{v(x)*v.T(x)}
    Then applies M_inv:

                v(x).T * M_inv * v(x)
    """
    def __init__(self, x_size, n):
        """
        x_size = vector_size, [x1 x2 ... xd]
        n = moment degree up to n
        """
        super(Q_real_M_2, self).__init__()
        self.n = n
        self.dim_veronese = int(comb(x_size + n, n))
        self.veroneses = []
        self.has_M_inv = False 
        self.M_inv = None
        self.build_M = True # Will be true when the autoencoder gets good reconstruction
        self.mybili = nn.Bilinear(self.dim_veronese, self.dim_veronese, 1, bias=None)
        
    def forward(self, x):
        if((not self.has_M_inv) and (self.build_M)):
            # We want only veronese maps to build the moment matrix once we have good reconstruction (self.build_M = True) !
            npoints, dims = x.size()
            v_x, _ = generate_veronese(x.view(dims, npoints).cuda(), self.n)
            self.veroneses.append(v_x.cpu())
        elif(self.has_M_inv):
            # Create the veronese map of z
            npoints, dims = x.size()
            print("npoints = ",npoints, " dims = ",dims)
            v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
            
            dim_veronese, BS = v_x.size()
            print("npoints = ",BS, " dims = ",dim_veronese)
            # first = torch.matmul(v_x.view(BS,1,dim_veronese), self.M_inv)
            # x = torch.matmul(first, v_x.view(BS, dim_veronese, 1))
            x = torch.matmul(
                torch.matmul(
                v_x.view(BS, 1, dim_veronese), self.M_inv),
                v_x.view(BS, dim_veronese, 1))
        return x

    def create_M(self):
        # This method should be created from outlise the class
        
        with torch.no_grad():
            n = len(self.veroneses)
            d, bs = self.veroneses[0].size()
            Mc = torch.tensor([])
            for i in range(0,n):
                print(i/n)
                #V = torch.cat([V, self.veroneses[i+1]], dim=1)
                V = self.veroneses[i]
                A = self.veroneses[i]
                A = A.view(bs,d,1)
                B = self.veroneses[i]
                B = B.view(bs,1,d)
                Mc_m = torch.matmul(A, B)
                # Mc_m = torch.bmm(V.view(bs,d,1), V.view(bs,1,d))
                Mc_m = torch.mean(Mc_m, dim=0)
                Mc = torch.cat([Mc,Mc_m.unsqueeze(0)])
                print(Mc.size())
            M = torch.mean(Mc,dim=0)
            self.M_inv = torch.inverse(M).cuda()
            # self.M_inv = torch.inverse(M)
            print(self.M_inv.size())
            self.has_M_inv = True
    
    def set_build_M(self):
        self.build_M = True
        self.create_M()


def generate_mixture(n):
       # MIXTURE OF GAUSSIANS
    # Set-up.
    np.random.seed(0x5eed)
    # Parameters of the mixture components
    norm_params = np.array([[0, 1],
                            [5, 0.3]])
                            # [9, 1.3]])
    n_components = norm_params.shape[0]
    # Weight of each component, in this case all of them are 1/3
    weights = np.ones(n_components, dtype=np.float32) / 2.0
    
    # A stream of indices from which to choose the component
    mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
    # y is the mixture sample
    x = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                    dtype=np.float32)
    return x

if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import torch
    from scipy.special import comb
    import matplotlib.pyplot as plt
    import scipy.stats as ss
    print(torch.get_default_dtype())

    expe = 2
    if expe==0:
        d = 64
        BS = 100
        # x = torch.tensor([1.0,2.0,3.0])
        x = torch.rand([d,BS])
        # x1 = torch.cat([torch.ones([1, BS]), x])
        n = 2 # degree of the polynomial, this will generate a moment matrix up to 2*n
        y, p = veronese_nk(x, n, if_cuda=False, if_coffiecnt=False)

        d = 64
        BS = 100
        # x = torch.tensor([1.0,2.0,3.0])
        mom = Q_real_M(d,n)
        for i in range(20):
            x = torch.rand([BS,d])
            _ = mom(x)
        
        # x1 = torch.cat([torch.ones([1, BS]), x])
        n = 2 # degree of the polynomial, this will generate a moment matrix up to 2*n
        # y, p = veronese_nk(x, n, if_cuda=False, if_coffiecnt=False)
        y, p = generate_veronese(x.cuda(),n)
        print("size y " + str(y.size()))
        
        dim_v = int(comb(d+n, n))
        print("dim_v : " + str(dim_v))
        V = torch.bmm(y.view(BS, dim_v ,1), y.view(BS, 1,dim_v))
        V = torch.mean(V,dim=0)
        U,S,V = torch.svd(V)
        print(S)
        M_inv = torch.inverse(V)
        v_x_test, _ = generate_veronese(torch.rand([1,64]).cuda(),n)
        Q = torch.matmul(torch.matmul(v_x_test.view(1,1,dim_v),M_inv), v_x_test.view(1,dim_v,1))
        print("Inlier: ", Q.item())
        v_x_test, _ = generate_veronese(torch.rand([64,1]).cuda()*20.0,n)
        Q = torch.matmul(torch.matmul(v_x_test.view(1,1,dim_v),M_inv), v_x_test.view(1,dim_v,1))
        print("Outlier: ",Q.item())
    elif expe==1:
        d = 1
        n = 3
        dim_v = int(comb(d+n, n))
        BS = dim_v - 1
        # x = torch.tensor([1.0,2.0,3.0])
        mom = Q_real_M(d,n)
        for i in range(20):
            print(i/20)
            x = torch.randn([BS,d])
            x = x.cuda()
            _ = mom(x)
        mom.create_M()
        # 0 inlier 1 outlier
        
        thr = 5*dim_v
        num_examples = dim_v - 1 ## numero magico si no no funciona 
        x_test = torch.randn([num_examples,d]).cuda()
        # Q = torch.matmul(torch.matmul(v_x_test.view(1,1,dim_v),M_inv), v_x_test.view(1,dim_v,1))
        Q = mom(x_test)
        print("Inlier: ", Q)
        predictions_out = torch.ge(Q,thr).view(num_examples)
        preds_in = torch.zeros(num_examples)
        preds_in[predictions_out] = 1
        gt_in = torch.zeros(num_examples)

        x_test = torch.randn([num_examples,d]).cuda()+20.0*torch.ones(num_examples, d).cuda()
        # Q = torch.matmul(torch.matmul(v_x_test.view(1,1,dim_v),M_inv), v_x_test.view(1,dim_v,1))
        Q = mom(x_test)
        predicions_out = torch.ge(Q,thr).view(num_examples)
        
        preds_out = torch.zeros(num_examples)
        preds_out[predicions_out] = 1
        gt_out = torch.ones(num_examples)
        preds = torch.cat([preds_in, preds_out], dim=0).numpy()
        preds = np.uint8(preds)

        gt = torch.cat([gt_in, gt_out], dim=0).numpy()
        gt = np.uint8(gt)

        tn, fp, fn, tp = confusion_matrix(gt, preds).ravel()
       
        p = tp/(tp+fp)
        
        r = tp/(tp+fn)

        print("Outlier: ", Q)
        f = 2*((p*r)/(p+r))
        print("F score = ", f)
    elif expe ==2:
        d = 1
        n = 4
        dim_v = int(comb(d+n, n))
        BS = 1
        mom = Q_real_M(d,n)
        samples_distr = torch.zeros(BS*4000,d)
        x = generate_mixture(4000)
        
        for i in range(4000):
            # y = torch.randn([BS,d])    
            y = torch.from_numpy(np.array(x[i])).view(BS,d)
            samples_distr[i*BS:i*BS+BS, :] = y
            y = y.cuda()
            _ = mom(y)
        mom.create_M()
        # 0 inlier 1 outlie 
        thr = 5*dim_v
        num_examples = 1 ## numero magico si no no funciona 
        x_lim = 7
        x_axis_len = 2*100*x_lim + 1
        x_axis = np.linspace(-x_lim, x_lim, x_axis_len)
        q_eval = torch.zeros(x_axis_len)
        count = 0
        for x in x_axis:
            x_test = torch.ones([num_examples,d]).cuda() * x       
            Q = mom(x_test)
            q_eval[count] = Q
            count=count+1
        np.save('samples_mixt_gauss_0_1_dim1_order_4_noROI'+str(n)+'.npy', samples_distr.numpy())
        np.save('q_eval__mixt_gauss_0_1_dim1_order_4_noROI'+str(n)+'.npy', q_eval.numpy())


        x_test = torch.randn([num_examples,d]).cuda()
        
        print("inlier value: ", x_test)
        Q = mom(x_test)
        print("Inlier: ", Q)
        predictions_out = torch.ge(Q,thr).view(num_examples)
        preds_in = torch.zeros(num_examples)
        preds_in[predictions_out] = 1
        gt_in = torch.zeros(num_examples)

        x_test = torch.randn([num_examples,d]).cuda()+20.0*torch.ones(num_examples, d).cuda()
        
        # Q = torch.matmul(torch.matmul(v_x_test.view(1,1,dim_v),M_inv), v_x_test.view(1,dim_v,1))
        Q = mom(x_test)
        predicions_out = torch.ge(Q,thr).view(num_examples)
        
        preds_out = torch.zeros(num_examples)
        preds_out[predicions_out] = 1
        gt_out = torch.ones(num_examples)
        preds = torch.cat([preds_in, preds_out], dim=0).numpy()
        preds = np.uint8(preds)

        gt = torch.cat([gt_in, gt_out], dim=0).numpy()
        gt = np.uint8(gt)

        tn, fp, fn, tp = confusion_matrix(gt, preds).ravel()
       
        p = tp/(tp+fp)
        
        r = tp/(tp+fn)

        print("Outlier: ", Q)
        f = 2*((p*r)/(p+r))
        print("F score = ", f)