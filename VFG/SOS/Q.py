import torch
import torch.nn as nn
import sys
sys.path.append("/home/ppalau/moments-vae/")
from SOS.veronese import generate_veronese as generate_veronese
from scipy.special import comb
import numpy as np

class Q(nn.Module):
    """
    This module is used to learn the inverse of the matrix of moments.
    Since Q(x) is low when evaluating the veronese map of an inlier and high for an outlier, we'll try
    to minimize:
                
                v(x).T * A * v(x)
    
    This class implements the operation above using torch.nn.Bilinear
    
    """
    def __init__(self, x_size, n):
        """
        x_size = vector_size, [x1 x2 ... xd]
        n = moment degree up to n
        """
        super(Q, self).__init__()
        self.n = n
        self.dim_veronese = int(comb(x_size + n, n))
        self.B = nn.Bilinear(self.dim_veronese, self.dim_veronese, 1, bias=None)
       
    def forward(self, x):
        npoints, dims = x.size()
        v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
        # v_x is (dim_veronese, BS), transpose it to have the batch dim at the beginning
        x = self.B(v_x.t_(), v_x)        
        return x

    def get_norm_of_B(self):
        one, rows, cols = self.B.weight.size()
        aux = self.B.weight.view(rows, cols)
        return torch.trace(torch.mm(aux.t(), aux))


class Bilinear_ATA(nn.Module):
    """
    This class implements the operation:
                
                x.T * A.T * A * x

    """
    def __init__(self, x_size):
        super(Bilinear_ATA, self).__init__()
        self.x_size = x_size # x_size is the same of dim_veronese
        self.A = torch.nn.Parameter(data=torch.rand(x_size,x_size), requires_grad=True)
    
    def forward(self, x):    
        # x represents the veronese, which will be of size (dim_veronese, BS)
        dim_veronese, BS = x.size()
        x = torch.matmul( 
            torch.matmul(
            torch.matmul(
                x.view(BS,1,dim_veronese), self.A.t()), 
                self.A),
                x.view(BS,dim_veronese,1)) 
        # The output will be of size BS, we resize it to be (BS,1)
        x = x.view(BS,1)
        return x


class Q_PSD(nn.Module):
    """
    This class implements the following operation:
                    
                v(x).T * A.T * A * v(x)  ,  where v(x) is the veronese map of x of order n 
    """
    def __init__(self, x_size, n):
        super(Q_PSD, self).__init__()
        self.n = n
        self.x_size = x_size
        self.dim_veronese = int(comb(x_size + n, n))
        self.B = Bilinear_ATA(self.dim_veronese)

    def forward(self, x):
        npoints, dims = x.size()
        v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
        # v_x is (dim_veronese, BS)
        x = self.B(v_x)        
        return x

    def get_norm_of_ATA(self):
        return torch.trace(torch.matmul(self.B.A.data.t(),self.B.A.data))


class Q_hinge_loss(nn.Module):
    """
    This loss is defined as follows:
        max(   0  ,  x - m   ) ; where x will be abs(vt(x) * A * v(x)) 
    """
    def __init__(self, order, dim):
        super(Q_hinge_loss, self).__init__()
        self.magic_Q = torch.tensor(comb(order+dim, dim))
    
    def forward(self, x):
        return torch.max(torch.zeros_like(x), x - (self.magic_Q.cuda(1) * torch.ones_like(x)))


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
        self.build_M = False # Will be true when the autoencoder gets good reconstruction

    def forward(self, x):
        if(not self.has_M_inv and self.build_M):
            # We want only veronese maps to build the moment matrix once we have good reconstruction (self.build_M = True) !
            npoints, dims = x.size()
            v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
            self.veroneses.append(v_x.cpu())
        elif(self.has_M_inv):
            # Create the veronese map of z
            npoints, dims = x.size()
            v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
            dim_veronese, BS = v_x.size()
            x = torch.matmul(
                torch.matmul(
                v_x.view(BS, 1, dim_veronese), self.M_inv),
                v_x.view(BS, dim_veronese, 1))
        return x

    def create_M(self):
        # This method should be created from outlise the class
        n = len(self.veroneses)
        d, bs = self.veroneses[0].size()
        V = self.veroneses[0]
        for i in range(0,n - 1 ):
            V = torch.cat([V, self.veroneses[i+1]], dim=1)
        V = torch.matmul(V.view(bs*n,d,1), V.view(bs*n,1,d))
        V = torch.mean(V,dim=0)
        self.M_inv = torch.inverse(V).cuda('cuda:2')
        self.has_M_inv = True
    
    def set_build_M(self):
        self.build_M = True


class Q_real_M_batches(nn.Module):
    """
    This module is in charge of 
        1. Building a moment matrix with the training samples (INLIERS)
            2. Applying the inverse of the empirically built moment matrix to discriminate outliers/inliers
    
    Basically, M = sum{v(x)*v.T(x)}
    Then applies M_inv:

                v(x).T * M_inv * v(x)
    NOTE: STILL IN MAINTENANCE
    """
    def __init__(self, x_size, n):
        """
        x_size = vector_size, [x1 x2 ... xd]
        n = moment degree up to n
        """
        super(Q_real_M_batches, self).__init__()
        self.n = n
        self.dim_veronese = int(comb(x_size + n, n))
        
        self.evaluation = False
        self.M_inv_copy = torch.eye(self.dim_veronese)
    def forward(self, x):
        if(not self.evaluation):
            # Create the veronese map of z
            npoints, dims = x.size()
            v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
            dim_veronese, BS = v_x.size()
            M_inv_temp = self.create_M(v_x).cuda(1)
            #TODO: Update Minv batch wise with sherman-morrisson techniques
            M_inv = ((self.M_inv_copy.cuda(1) + M_inv_temp)*0.5)
            
            del M_inv_temp
            torch.cuda.empty_cache()
            
            x = torch.matmul(
                torch.matmul(
                v_x.view(BS, 1, dim_veronese), M_inv),
                v_x.view(BS, dim_veronese, 1))
            self.M_inv_copy = M_inv.cpu().detach().clone()
        else:
            npoints, dims = x.size()
            v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
            dim_veronese, BS = v_x.size()
            x = torch.matmul(
                torch.matmul(
                v_x.view(BS, 1, dim_veronese), self.M_inv_copy.cuda(1)),
                v_x.view(BS, dim_veronese, 1))

        return x

    def create_M(self, v_x):
        d, bs = v_x.size()
        V = torch.matmul(v_x.view(bs,d,1), v_x.view(bs,1,d))
        V = torch.mean(V,dim=0)
        M_inv = torch.inverse(V).cuda(1)
        return M_inv


    def set_eval(self, b):
        assert isinstance(b,bool),"b must be boolean"
        self.evaluation = b

class MyBilinear(nn.Module):
    """
    Class created to solve the bug in Q, which used bilinear operation of PyTorch and seemed to work bad.
    It implements the operation:
    
                x.T * A * x
    """
    def __init__(self, x_size):
        super(MyBilinear, self).__init__()
        self.x_size = x_size # x_size is the same of dim_veronese
        sqrt_k = np.sqrt(1/self.x_size)
        self.A = torch.nn.Parameter(data=torch.rand(x_size,x_size)*sqrt_k -sqrt_k*0.5, requires_grad=True)
    
    def forward(self, x):    
        # x represents the veronese, which will be of size (dim_veronese, BS)
        dim_veronese, BS = x.size()
        x = torch.matmul( 
            torch.matmul(
                x.view(BS,1,dim_veronese),self.A),
                x.view(BS,dim_veronese,1))
        # The output will be of size BS, we resize it to be (BS,1)
        x = x.view(BS,1) 
        return x


class Q_MyBilinear(nn.Module):
    """
    This class implements the following operation
        
                v(x).T * A * v(x) 
    """
    def __init__(self, x_size, n):
        """
        x_size = vector_size, [x1 x2 ... xd]
        n = moment degree up to n
        """
        super(Q_MyBilinear, self).__init__()
        self.n = n
        self.dim_veronese = int(comb(x_size+n, n))
        self.B = MyBilinear(self.dim_veronese)

    def forward(self, x):
        npoints, dims = x.size()
        v_x, _ = generate_veronese(x.view(dims, npoints), self.n)
        # v_x is (dim_veronese, BS), transpose it to have the batch dim at the beginning
        x = self.B(v_x)        
        return x

    def get_norm_of_B(self):
        rows, cols = self.B.A.data.size()
        aux = self.B.A.data.view(rows, cols)
        return torch.trace(torch.mm(aux.t(), aux))