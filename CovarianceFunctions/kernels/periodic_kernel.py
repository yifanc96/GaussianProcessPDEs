import jax.numpy as jnp
from jax import grad, jvp, hessian


def kappa(x,y,d,sigma, p = 1):
    # dist = jnp.sqrt((x-y)**2 + eps)
    # val = jnp.exp((jnp.sum(jnp.cos(p*jnp.pi*dist)-1))/sigma**2)
    
    dist = jnp.sum((jnp.cos(2*p*jnp.pi*x)-jnp.cos(2*p*jnp.pi*y))**2 + (jnp.sin(2*p*jnp.pi*x)-jnp.sin(2*p*jnp.pi*y))**2) # simplify it later!
    val = jnp.exp(-dist/sigma**2)
    return val


def D_wy_kappa(x,y,d, sigma,w):
    _, val = jvp(lambda y: kappa(x,y,d,sigma),(y,),(w,))
    return val

def Delta_y_kappa(x,y,d,sigma):
    val = jnp.trace(hessian(lambda y: kappa(x,y,d,sigma))(y))
    return val

def D_wx_kappa(x,y,d, sigma,w):
    _, val = jvp(lambda x: kappa(x,y,d,sigma),(x,),(w,))
    return val

# Dx vector
def D_x_kappa(x,y,d, sigma):
    val = grad(lambda x: kappa(x,y,d,sigma))(x)
    return val

def D_wx_D_wy_kappa(x,y,d,sigma,wx,wy):
    _, val = jvp(lambda x: D_wy_kappa(x,y,d,sigma,wy),(x,),(wx,))
    return val

# # DxDwy vector
def D_x_D_wy_kappa(x,y,d,sigma,wy):
    val = grad(lambda x: D_wy_kappa(x,y,d,sigma,wy))(x)
    return val

def D_wx_Delta_y_kappa(x,y, d,sigma,w):
    val = jnp.trace(hessian(lambda y: D_wx_kappa(x,y,d, sigma,w))(y))
    return val

# # Delta
def Delta_x_kappa(x,y,d,sigma):
    val = jnp.trace(hessian(lambda x: kappa(x,y,d, sigma))(x))
    return val

def Delta_x_D_wy_kappa(x,y, d,sigma,w):
    val = jnp.trace(hessian(lambda x: D_wy_kappa(x,y,d, sigma,w))(x))
    return val

def Delta_x_Delta_y_kappa(x,y,d,sigma):
    val = jnp.trace(hessian(lambda x: Delta_y_kappa(x,y,d, sigma))(x))
    return val

# high order derivatives
def D_wy_Delta_y_kappa(x,y,d, sigma,w):
    _, val = jvp(lambda y: Delta_y_kappa(x,y,d,sigma),(y,),(w,))
    return val

def D_wx_Delta_x_kappa(x,y,d, sigma,w):
    _, val = jvp(lambda x: Delta_x_kappa(x,y,d,sigma),(x,),(w,))
    return val

def D_wx_Delta_x_D_wy_kappa(x,y,d,sigma,wx,wy):
    _, val = jvp(lambda y: D_wx_Delta_x_kappa(x,y,d, sigma,wx),(y,),(wy,))
    return val

def D_wx_D_wy_Delta_y_kappa(x,y,d,sigma,wx,wy):
    _, val = jvp(lambda x: D_wy_Delta_y_kappa(x,y,d,sigma,wy),(x,),(wx,))
    return val

def D_wx_Delta_x_D_wy_Delta_y_kappa(x,y,d,sigma,wx,wy):
    val = jnp.trace(hessian(lambda x: D_wx_D_wy_Delta_y_kappa(x,y,d,sigma,wx,wy))(x))
    return val

# test
# x = jnp.array([0.0,0.0])
# y = jnp.array([0.0,0.0])
# w = jnp.array([1.0,1.0])
# d = 2
# sigma = 0.2
# # print(D_wx_D_wy_kappa(x,y,d,sigma,w,w))
# print(D_wx_Delta_x_D_wy_Delta_y_kappa(x,y,d,sigma,w,w))




