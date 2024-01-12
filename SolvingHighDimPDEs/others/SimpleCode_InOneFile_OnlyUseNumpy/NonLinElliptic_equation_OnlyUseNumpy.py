
# numpy
import numpy as np
from numpy import random 

import logging
from time import time

# when using numpy for vectorization, must pay attention to a vector being row or column type; very important

def kappa(x,y,d,sigma):
    dist2 = np.sum((x-y)**2,axis=-1,keepdims=True)
    return np.exp(-dist2/(2*sigma**2))

def Delta_y_kappa(x,y,d,sigma):
    dist2 = np.sum((x-y)**2,axis=-1,keepdims=True)
    val = (-d*(sigma**2)+dist2)/(sigma**4)*np.exp(-dist2/(2*sigma**2))
    return val

def Delta_x_kappa(x,y,d,sigma):
    dist2 = np.sum((x-y)**2,axis=-1,keepdims=True)
    val = (-d*(sigma**2)+dist2)/(sigma**4)*np.exp(-dist2/(2*sigma**2))
    return val


def Delta_x_Delta_y_kappa(x,y,d, sigma):
    dist2 = np.sum((x-y)**2,axis=-1,keepdims=True)
    val = ((sigma**4)*d*(2+d)-2*(sigma**2)*(2+d)*dist2+dist2**2)/(sigma**8)*np.exp(-dist2/(2*sigma**2))
    return val

def get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma):
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*wy1*Delta_y_kappa(x,y,d,sigma) + wy0*wx1*Delta_x_kappa(x,y,d,sigma) + wx1*wy1*Delta_x_Delta_y_kappa(x,y,d,sigma)

def get_GNkernel_train_boundary(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)

def get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)


def assembly_Theta(X_domain, X_boundary, w0, w1, sigma):
    # X_domain, dim: N_domain*d; 
    # w0 col vec: coefs of Diracs, dim: N_domain; 
    # w1 coefs of Laplacians, dim: N_domain
    
    N_domain,d = np.shape(X_domain)
    N_boundary,_ = np.shape(X_boundary)
    Theta = np.zeros((N_domain+N_boundary,N_domain+N_boundary))
    
    XdXd0 = np.reshape(np.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = np.tile(X_domain,(N_domain,1))
    
    XbXd0 = np.reshape(np.tile(X_boundary,(1,N_domain)),(-1,d))
    XbXd1 = np.tile(X_domain,(N_boundary,1))
    
    XbXb0 = np.reshape(np.tile(X_boundary,(1,N_boundary)),(-1,d))
    XbXb1 = np.tile(X_boundary,(N_boundary,1))
    
    arr_wx0 = np.reshape(np.tile(w0,(1,N_domain)),(-1,1))
    arr_wx1 = np.reshape(np.tile(w1,(1,N_domain)),(-1,1))
    arr_wy0 = np.tile(w0,(N_domain,1))
    arr_wy1 = np.tile(w1,(N_domain,1))
    
    arr_wy0_bd = np.tile(w0,(N_boundary,1))
    arr_wy1_bd = np.tile(w1,(N_boundary,1))
    
    val = get_GNkernel_train(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wy0,arr_wy1,d,sigma)
    Theta[:N_domain,:N_domain] = np.reshape(val, (N_domain,N_domain))
    
    val = get_GNkernel_train_boundary(XbXd0,XbXd1,arr_wy0_bd,arr_wy1_bd,d,sigma)
    Theta[N_domain:,:N_domain] = np.reshape(val, (N_boundary,N_domain))
    Theta[:N_domain,N_domain:] = np.transpose(np.reshape(val, (N_boundary,N_domain)))
    
    val = kappa(XbXb0, XbXb1,d,sigma)
    Theta[N_domain:,N_domain:] = np.reshape(val, (N_boundary, N_boundary))
    return Theta
    
def assembly_Theta_value_predict(X_infer, X_domain, X_boundary, w0, w1, sigma):
    N_infer, d = np.shape(X_infer)
    N_domain, _ = np.shape(X_domain)
    N_boundary, _ = np.shape(X_boundary)
    Theta = np.zeros((N_infer,N_domain+N_boundary))
    
    XiXd0 = np.reshape(np.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = np.tile(X_domain,(N_infer,1))
    
    XiXb0 = np.reshape(np.tile(X_infer,(1,N_boundary)),(-1,d))
    XiXb1 = np.tile(X_boundary,(N_infer,1))
    
    arr_wy0 = np.tile(w0,(N_infer,1))
    arr_wy1 = np.tile(w1,(N_infer,1))
    
    val = get_GNkernel_val_predict(XiXd0,XiXd1,arr_wy0,arr_wy1,d,sigma)
    Theta[:N_infer,:N_domain] = np.reshape(val, (N_infer,N_domain))
    
    val = kappa(XiXb0, XiXb1,d,sigma)
    Theta[:N_infer,N_domain:] = np.reshape(val, (N_infer,N_boundary))
    return Theta

def GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = 4):
    N_domain, d = np.shape(X_domain)
    sol = sol_init
    rhs_f = f(X_domain)[:,np.newaxis]
    bdy_g = g(X_boundary)[:,np.newaxis]
    time_begin = time()
    for i in range(GN_step):
        w1 = -np.ones((N_domain,1))
        w0 = alpha*m*(sol**(m-1))
        Theta_train = assembly_Theta(X_domain, X_boundary, w0, w1, sigma)
        Theta_test = assembly_Theta_value_predict(X_domain, X_domain, X_boundary, w0, w1, sigma)
        rhs = rhs_f + alpha*(m-1)*(sol**m)
        rhs = np.concatenate((rhs, bdy_g), axis = 0)
        sol = Theta_test @ (np.linalg.solve(Theta_train + nugget*np.diag(np.diag(Theta_train)),rhs))
        total_mins = (time() - time_begin) / 60
        logging.info(f'[Timer] GP iteration {i+1}/{GN_step}, finished in {total_mins:.2f} minutes')
    return sol

def sample_points(N_domain, N_boundary, d, choice = 'random'):
    X_domain = np.zeros((N_domain,d))
    X_boundary = np.zeros((N_boundary,d))
    
    X_domain = np.random.randn(N_domain,d)  # N_domain*d
    X_domain /= np.linalg.norm(X_domain, axis=1)[:,np.newaxis] # the divisor is of N_domain*1
    random_radii = np.random.rand(N_domain,1) ** (1/d)
    X_domain *= random_radii
    
    X_boundary = np.random.randn(N_boundary,d)
    X_boundary /= np.linalg.norm(X_boundary, axis=1)[:,np.newaxis]
    return X_domain, X_boundary


def set_random_seeds(random_seed):
    random.seed(random_seed)
    
def u(x):
    return np.sin(np.sum(x,axis=-1))

def f(x):
    return d*np.sin(np.sum(x,axis=-1))+alpha*(u(x)**m)

def g(x):
    return u(x)

if __name__ == '__main__':
    ## get argument parser
    logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
    
    alpha = 1.0
    m = 3
    logging.info(f"[Equation] alpha: {alpha}, m: {m}")
    
    randomseed = 9999
    set_random_seeds(randomseed)
    logging.info(f"[Seeds] random seeds: {randomseed}")

    d = 10
    N_domain = 1000
    N_boundary = 200
    X_domain, X_boundary = sample_points(N_domain, N_boundary, d, choice = 'random')
    
    sol_init = np.random.randn(N_domain,1)
    ratio = 0.25
    sigma = ratio*np.sqrt(d)
    nugget = 1e-10
    GN_step = 4

    logging.info(f'GN step: {GN_step}, d: {d}, sigma: {sigma}, number of points: N_domain {N_domain}, N_boundary {N_boundary}, kernel: Gaussian, nugget: {nugget}')
    
    sol = GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = GN_step)

    logging.info('[Calculating errs at collocation points ...]')
    sol_truth = u(X_domain)[:,np.newaxis]
    err = abs(sol-sol_truth)
    err_2 = np.linalg.norm(err,'fro')/(onp.sqrt(N_domain))
    err_inf = np.max(err)
    logging.info(f'[L infinity error] {err_inf}')
    logging.info(f'[L2 error] {err_2}')