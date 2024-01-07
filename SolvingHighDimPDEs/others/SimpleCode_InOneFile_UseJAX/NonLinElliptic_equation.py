import jax.numpy as jnp
from jax import grad, vmap, hessian, jit

from jax.config import config; 
config.update("jax_enable_x64", True)

# numpy
import numpy as onp
from numpy import random 

import logging
from time import time


def kappa(x,y,d,sigma):
    dist2 = sum((x-y)**2)
    return jnp.exp(-dist2/(2*sigma**2))

def Delta_y_kappa(x,y,d,sigma):
    dist2 = sum((x-y)**2)
    val = (-d*(sigma**2)+dist2)/(sigma**4)*jnp.exp(-dist2/(2*sigma**2))
    return val

def Delta_x_kappa(x,y,d,sigma):
    dist2 = sum((x-y)**2)
    val = (-d*(sigma**2)+dist2)/(sigma**4)*jnp.exp(-dist2/(2*sigma**2))
    return val


def Delta_x_Delta_y_kappa(x,y,d, sigma):
    dist2 = sum((x-y)**2)
    val = ((sigma**4)*d*(2+d)-2*(sigma**2)*(2+d)*dist2+dist2**2)/(sigma**8)*jnp.exp(-dist2/(2*sigma**2))
    return val
@jit
def get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma):
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*wy1*Delta_y_kappa(x,y,d,sigma) + wy0*wx1*Delta_x_kappa(x,y,d,sigma) + wx1*wy1*Delta_x_Delta_y_kappa(x,y,d,sigma)
@jit
def get_GNkernel_train_boundary(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)
@jit
def get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)


def assembly_Theta(X_domain, X_boundary, w0, w1, sigma):
    # X_domain, dim: N_domain*d; 
    # w0 col vec: coefs of Diracs, dim: N_domain; 
    # w1 coefs of Laplacians, dim: N_domain
    
    N_domain,d = onp.shape(X_domain)
    N_boundary,_ = onp.shape(X_boundary)
    Theta = onp.zeros((N_domain+N_boundary,N_domain+N_boundary))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    XbXd0 = onp.reshape(onp.tile(X_boundary,(1,N_domain)),(-1,d))
    XbXd1 = onp.tile(X_domain,(N_boundary,1))
    
    XbXb0 = onp.reshape(onp.tile(X_boundary,(1,N_boundary)),(-1,d))
    XbXb1 = onp.tile(X_boundary,(N_boundary,1))
    
    arr_wx0 = onp.reshape(onp.tile(w0,(1,N_domain)),(-1,1))
    arr_wx1 = onp.reshape(onp.tile(w1,(1,N_domain)),(-1,1))
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    
    arr_wy0_bd = onp.tile(w0,(N_boundary,1))
    arr_wy1_bd = onp.tile(w1,(N_boundary,1))
    
    val = vmap(lambda x,y,wx0,wx1,wy0,wy1: get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma))(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wy0,arr_wy1)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_train_boundary(x,y,wy0,wy1,d,sigma))(XbXd0,XbXd1,arr_wy0_bd,arr_wy1_bd)
    Theta[N_domain:,:N_domain] = onp.reshape(val, (N_boundary,N_domain))
    Theta[:N_domain,N_domain:] = onp.transpose(onp.reshape(val, (N_boundary,N_domain)))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XbXb0, XbXb1)
    Theta[N_domain:,N_domain:] = onp.reshape(val, (N_boundary, N_boundary))
    return Theta
    
def assembly_Theta_value_predict(X_infer, X_domain, X_boundary, w0, w1, sigma):
    N_infer, d = onp.shape(X_infer)
    N_domain, _ = onp.shape(X_domain)
    N_boundary, _ = onp.shape(X_boundary)
    Theta = onp.zeros((N_infer,N_domain+N_boundary))
    
    XiXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = onp.tile(X_domain,(N_infer,1))
    
    XiXb0 = onp.reshape(onp.tile(X_infer,(1,N_boundary)),(-1,d))
    XiXb1 = onp.tile(X_boundary,(N_infer,1))
    
    arr_wy0 = onp.tile(w0,(N_infer,1))
    arr_wy1 = onp.tile(w1,(N_infer,1))
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XiXb0, XiXb1)
    Theta[:N_infer,N_domain:] = onp.reshape(val, (N_infer,N_boundary))
    return Theta

def GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = 4):
    N_domain, d = onp.shape(X_domain)
    sol = sol_init
    rhs_f = vmap(f)(X_domain)[:,onp.newaxis]
    bdy_g = vmap(g)(X_boundary)[:,onp.newaxis]
    time_begin = time()
    for i in range(GN_step):
        w1 = -onp.ones((N_domain,1))
        w0 = alpha*m*(sol**(m-1))
        Theta_train = assembly_Theta(X_domain, X_boundary, w0, w1, sigma)
        Theta_test = assembly_Theta_value_predict(X_domain, X_domain, X_boundary, w0, w1, sigma)
        rhs = rhs_f + alpha*(m-1)*(sol**m)
        rhs = onp.concatenate((rhs, bdy_g), axis = 0)
        sol = Theta_test @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
        total_mins = (time() - time_begin) / 60
        logging.info(f'[Timer] GP iteration {i+1}/{GN_step}, finished in {total_mins:.2f} minutes')
    return sol

def sample_points(N_domain, N_boundary, d, choice = 'random'):
    X_domain = onp.zeros((N_domain,d))
    X_boundary = onp.zeros((N_boundary,d))
    
    X_domain = onp.random.randn(N_domain,d)  # N_domain*d
    X_domain /= onp.linalg.norm(X_domain, axis=1)[:,onp.newaxis] # the divisor is of N_domain*1
    random_radii = onp.random.rand(N_domain,1) ** (1/d)
    X_domain *= random_radii
    
    X_boundary = onp.random.randn(N_boundary,d)
    X_boundary /= onp.linalg.norm(X_boundary, axis=1)[:,onp.newaxis]
    return X_domain, X_boundary


def set_random_seeds(random_seed):
    random.seed(random_seed)
    
def u(x):
    return jnp.sin(jnp.sum(x))
def f(x):
    return -jnp.trace(hessian(u)(x))+alpha*(u(x)**m)
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

    d = 4
    N_domain = 1000
    N_boundary = 200
    X_domain, X_boundary = sample_points(N_domain, N_boundary, d, choice = 'random')
    
    sol_init = onp.random.randn(N_domain,1)
    ratio = 0.25
    sigma = ratio*onp.sqrt(d)
    nugget = 1e-10
    GN_step = 4

    logging.info(f'GN step: {GN_step}, d: {d}, sigma: {sigma}, number of points: N_domain {N_domain}, N_boundary {N_boundary}, kernel: Gaussian, nugget: {nugget}')
    
    sol = GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = GN_step)

    logging.info('[Calculating errs at collocation points ...]')
    sol_truth = vmap(u)(X_domain)[:,onp.newaxis]
    err = abs(sol-sol_truth)
    err_2 = onp.linalg.norm(err,'fro')/(onp.sqrt(N_domain))
    err_inf = onp.max(err)
    logging.info(f'[L infinity error] {err_inf}')
    logging.info(f'[L2 error] {err_2}')