import jax.numpy as jnp
from jax import jvp, hessian, grad

def kappa(x1, x2, y1, y2, sigma):
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    return jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))

# cover derivatives over x1 and y1

def D_wy1_kappa(x1, x2, y1, y2, sigma, w):
    _, val = jvp(lambda y1: kappa(x1, x2, y1, y2, sigma),(y1,),(w,))
    return val

def Delta_y1_kappa(x1, x2, y1, y2, sigma):
    val = jnp.trace(hessian(lambda y1: kappa(x1, x2, y1, y2, sigma))(y1))
    return val

def D_wx1_kappa(x1, x2, y1, y2, sigma,w):
    _, val = jvp(lambda x1: kappa(x1, x2, y1, y2, sigma),(x1,),(w,))
    return val

# Dx vector
def D_x1_kappa(x1, x2, y1, y2, sigma):
    val = grad(lambda x1: kappa(x1, x2, y1, y2, sigma))(x1)
    return val

def D_wx1_D_wy1_kappa(x1, x2, y1, y2, sigma,wx1,wy1):
    _, val = jvp(lambda x1: D_wy1_kappa(x1, x2, y1, y2, sigma,wy1),(x1,),(wx1,))
    return val

# # DxDwy1 vector
def D_x1_D_wy1_kappa(x1, x2, y1, y2, sigma,wy1):
    val = grad(lambda x1: D_wy1_kappa(x1, x2, y1, y2, sigma,wy1))(x1)
    return val

def D_wx1_Delta_y1_kappa(x1, x2, y1, y2, sigma,w):
    val = jnp.trace(hessian(lambda y1: D_wx1_kappa(x1, x2, y1, y2, sigma,w))(y1))
    return val

# # Delta
def Delta_x1_kappa(x1, x2, y1, y2, sigma):
    val = jnp.trace(hessian(lambda x1: kappa(x1, x2, y1, y2, sigma))(x1))
    return val

def Delta_x1_D_wy1_kappa(x1, x2, y1, y2, sigma,w):
    val = jnp.trace(hessian(lambda x: D_wy1_kappa(x1, x2, y1, y2, sigma,w))(x1))
    return val

def Delta_x1_Delta_y1_kappa(x1, x2, y1, y2, sigma):
    val = jnp.trace(hessian(lambda x1: Delta_y1_kappa(x1, x2, y1, y2, sigma))(x1))
    return val

