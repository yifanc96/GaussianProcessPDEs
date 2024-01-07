import jax.numpy as jnp
from jax import vmap, jit

from jax.config import config; 
config.update("jax_enable_x64", True)

# numpy
import numpy as onp

import logging
# up to 2nd derivatives
        
def kappa(x,y,d,sigma):
    dist2 = sum((x-y)**2)
    return jnp.exp(-dist2/(2*sigma**2))

def D_wy_kappa(x,y,d, sigma,w):
    dist2 = sum((x-y)**2)
    val = -sum(w*(y-x))/(sigma**2)*jnp.exp(-dist2/(2*sigma**2))
    return val

def Delta_y_kappa(x,y,d,sigma):
    dist2 = sum((x-y)**2)
    val = (-d*(sigma**2)+dist2)/(sigma**4)*jnp.exp(-dist2/(2*sigma**2))
    return val

def D_wx_kappa(x,y,d, sigma,w):
    dist2 = sum((x-y)**2)
    val = -sum(w*(x-y))/(sigma**2)*jnp.exp(-dist2/(2*sigma**2))
    return val

# Dx vector
def D_x_kappa(x,y,d, sigma):
    dist2 = sum((x-y)**2)
    val = -(x-y)/(sigma**2)*jnp.exp(-dist2/(2*sigma**2))
    return val

def D_wx_D_wy_kappa(x,y,d,sigma,wx,wy):
    dist2 = sum((x-y)**2)
    val = (sum(wx*wy)/(sigma**2)+sum(wx*(x-y))*sum(wy*(y-x))/(sigma**4))*jnp.exp(-dist2/(2*sigma**2))
    return val

# DxDwy vector
def D_x_D_wy_kappa(x,y,d,sigma,wy):
    dist2 = sum((x-y)**2)
    val = (wy/(sigma**2)+(x-y)*sum(wy*(y-x))/(sigma**4))*jnp.exp(-dist2/(2*sigma**2))
    return val

def D_wx_Delta_y_kappa(x,y, d,sigma,w):
    dist2 = sum((x-y)**2)
    val = sum(w*(x-y))*((sigma**2)*(2+d)-dist2)/(sigma**6)*jnp.exp(-dist2/(2*sigma**2))
    return val


def Delta_x_kappa(x,y,d,sigma):
    dist2 = sum((x-y)**2)
    val = (-d*(sigma**2)+dist2)/(sigma**4)*jnp.exp(-dist2/(2*sigma**2))
    return val

def Delta_x_D_wy_kappa(x,y, d,sigma,w):
    dist2 = sum((x-y)**2)
    val = sum(w*(y-x))*((sigma**2)*(2+d)-dist2)/(sigma**6)*jnp.exp(-dist2/(2*sigma**2))
    return val

def Delta_x_Delta_y_kappa(x,y,sigma):
    dist2 = sum((x-y)**2)
    val = ((sigma**4)*d*(2+d)-2*(sigma**2)*(2+d)*dist2+dist2**2)/(sigma**8)*jnp.exp(-dist2/(2*sigma**2))
    return val
@jit
def get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma):
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*D_wy_kappa(x,y,d, sigma,wy1) + wy0* D_wx_kappa(x,y,d, sigma,wx1) + D_wx_D_wy_kappa(x,y,d,sigma,wx1,wy1)
@jit
def get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + D_wy_kappa(x,y,d, sigma,wy1)
@jit
def get_GNkernel_grad_predict(x,y,wy0,wy1,d,sigma):
    return wy0*D_x_kappa(x,y,d, sigma) + D_x_D_wy_kappa(x,y,d,sigma,wy1)

def g(x):
    return jnp.log(1/2+1/2*sum(x**2))

def assembly_Theta(X_domain, w0, w1, sigma):
    # X_domain, dim: N_domain*d; 
    # w0 col vec: coefs of Diracs, dim: N_domain; 
    # w1 coefs of gradients, dim: N_domain*d
    
    N_domain,d = onp.shape(X_domain)
    Theta = onp.zeros((N_domain,N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    arr_wx0 = onp.reshape(onp.tile(w0,(1,N_domain)),(-1,1))
    arr_wx1 = onp.reshape(onp.tile(w1,(1,N_domain)),(-1,d))
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    
    val = vmap(lambda x,y,wx0,wx1,wy0,wy1: get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma))(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wy0,arr_wy1)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    return Theta
    
def assembly_Theta_value_and_grad_predict(X_infer, X_domain, w0, w1, sigma):
    N_infer, d = onp.shape(X_infer)
    N_domain, _ = onp.shape(X_domain)
    Theta = onp.zeros((N_infer*(d+1),N_domain))
    
    XiXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = onp.tile(X_domain,(N_infer,1))
    
    arr_wy0 = onp.tile(w0,(N_infer,1))
    arr_wy1 = onp.tile(w1,(N_infer,1))
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_domain,N_domain))

    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_grad_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[N_infer:,:N_domain] = onp.reshape(val,(N_infer*d,N_domain))
    return Theta
    
def assembly_Theta_stanGP(X_domain,sigma):
    N_domain,d = onp.shape(X_domain)
    Theta = onp.zeros((N_domain,N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XdXd0,XdXd1)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    return Theta
    
def assembly_Theta_predict_value_and_grad_stanGP(X_infer, X_domain,sigma):
    N_infer,d = onp.shape(X_infer)
    N_domain = onp.shape(X_domain)[0]
    Theta = onp.zeros((N_infer*(d+1),N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_infer,1))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XdXd0,XdXd1)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    val = vmap(lambda x,y: D_x_kappa(x,y,d,sigma))(XdXd0,XdXd1)
    Theta[N_infer:,:N_domain] = onp.reshape(val,(N_infer*d,N_domain))
    return Theta

def generate_path(X_init, N_domain, dt, T):
    if onp.ndim(X_init)==1: X_init = X_init[onp.newaxis,:]
    _,d = onp.shape(X_init)
    Nt = int(T/dt)+1
    arr_X = onp.zeros((Nt,N_domain,d))
    arr_X[0,:,:] = X_init
    rdn = onp.random.normal(0, 1, (Nt-1, N_domain,d))
    for i in range(Nt-1):
        arr_X[i+1,:,:] = arr_X[i,:,:] + onp.sqrt(2*dt)*rdn[i,:,:]
    return arr_X

def one_step_iteration(V_future, X_future, X_now, dt, sigma, nugget, GN_step):
    N_domain = onp.shape(X_now)[0]
    Theta_train = assembly_Theta_stanGP(X_future,sigma)
    Theta_infer = assembly_Theta_predict_value_and_grad_stanGP(X_now, X_future,sigma)
    
    V_val_n_grad = Theta_infer @ (onp.linalg.solve(Theta_train + nugget*onp.eye(onp.shape(Theta_train)[0]),V_future))
    w0 = onp.ones((N_domain,1))
    for i in range(GN_step):
        # get grad V_{old}
        V_old = V_val_n_grad[:N_domain]
        logger.info(f'  [logs] GN step: {i}, and sol val at the 1st point {V_old[0]}')
        V_old_grad = onp.reshape(V_val_n_grad[N_domain:],(N_domain,d))
        
        w1 = 2*V_old_grad+(X_future-X_now)
        Theta_train = assembly_Theta(X_now, w0, w1, sigma)
        Theta_infer = assembly_Theta_value_and_grad_predict(X_now, X_now, w0, w1, sigma)
        rhs = V_future + onp.sum(V_old_grad**2,axis=1)*dt
        V_val_n_grad = Theta_infer @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
    
    return V_val_n_grad[:N_domain]


def GPsolver(X_init, N_domain, dt, T, sigma, nugget, GN_step = 4):
    if onp.ndim(X_init)==1: X_init = X_init[onp.newaxis,:]
    _,d = onp.shape(X_init)
    Nt = int(T/dt)+1
    arr_X = generate_path(X_init, N_domain, dt, T)
    V = onp.zeros((Nt,N_domain))
    V[-1,:] = vmap(g)(arr_X[-1,:,:])
    
    # solve V[-i-1,:] from V[-i,:]
    for i in onp.arange(1,Nt):
        logger.info(f'[Time marching] at iteration {i}/{Nt-1}, solving eqn at time t = {t:.2f}')
        V[-i-1,:] = one_step_iteration(V[-i,:], arr_X[-i,:,:], arr_X[-i-1,:,:], dt, sigma, nugget, GN_step)
        t = (Nt-i-1)*dt
    return V


d = 100
X_init = onp.zeros((1,d))
N_domain = 1000
dt = 1e-2
T = 1

ratio = 500
sigma = ratio*onp.sqrt(d)
nugget = 0.04
GN_step = 4

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()]
        )
logger=logging.getLogger()

logger.info(f'GN step: {GN_step}, d: {d}, sigma: {sigma}, points: {N_domain}')
V = GPsolver(X_init, N_domain, dt, T, sigma, nugget, GN_step)