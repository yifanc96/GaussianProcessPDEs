import jax.numpy as jnp
from jax import grad, jvp, hessian

eps = 1e-8

def kappa(x,y,d,sigma):
    dist = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    val = (1+jnp.sqrt(3)*dist/sigma) * jnp.exp(-jnp.sqrt(3)*dist/sigma)
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



# test
# x = jnp.array([0.0,0.0])
# y = jnp.array([0.0,0.0])
# w = jnp.array([1.0,1.0])
# d = 2
# sigma = 0.2
# print(D_wx_D_wy_kappa(x,y,d,sigma,w,w))




