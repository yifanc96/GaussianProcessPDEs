import jax.numpy as jnp

def kappa(x,y,d,sigma):
    dist2 = jnp.sum((x-y)**2)
    return 1/(dist2/(2*sigma**2)+1)

def D_wy_kappa(x,y,d, sigma,w):
    dist2 = jnp.sum((x-y)**2)
    val = -jnp.sum(w*(y-x))/(sigma**2 * (dist2/(2*sigma**2)+1)**2)
    return val

def Delta_y_kappa(x,y,d,sigma):
    dist2 = jnp.sum((x-y)**2)
    val = -4*sigma**2*(2*(sigma**2)*d+(-4+d)*dist2)/(2*(sigma**2)+dist2)**3
    return val

def D_wx_kappa(x,y,d, sigma,w):
    dist2 = jnp.sum((x-y)**2)
    val = -jnp.sum(w*(x-y))/(sigma**2 * (dist2/(2*sigma**2)+1)**2)
    return val

# Dx vector
def D_x_kappa(x,y,d, sigma):
    dist2 = jnp.sum((x-y)**2)
    val = -(x-y)/(sigma**2 * (dist2/(2*sigma**2)+1)**2)
    return val

def D_wx_D_wy_kappa(x,y,d,sigma,wx,wy):
    dist2 = jnp.sum((x-y)**2)
    val = jnp.sum(wx*wy)/(sigma**2 * (dist2/(2*sigma**2)+1)**2) + 2*jnp.sum(wx*(x-y))*jnp.sum(wy*(y-x))/(sigma**4 * (dist2/(2*sigma**2)+1)**3) 
    return val

# DxDwy vector
def D_x_D_wy_kappa(x,y,d,sigma,wy):
    dist2 = jnp.sum((x-y)**2)
    val = wy/(sigma**2 * (dist2/(2*sigma**2)+1)**2) + 2*(x-y)*(jnp.sum(wy*(y-x)))/(sigma**4 * (dist2/(2*sigma**2)+1)**3) 
    return val

def D_wx_Delta_y_kappa(x,y, d,sigma,w):
    dist2 = jnp.sum((x-y)**2)
    val = jnp.sum(w*(x-y))*(16*sigma**2)*(2*sigma**2*(2+d)+(-4+d)*dist2)/(2*sigma**2+dist2)**4
    return val

# Delta
def Delta_x_kappa(x,y,d,sigma):
    dist2 = jnp.sum((x-y)**2)
    val = -4*sigma**2*(2*(sigma**2)*d+(-4+d)*dist2)/(2*(sigma**2)+dist2)**3
    return val

def Delta_x_D_wy_kappa(x,y, d,sigma,w):
    dist2 = jnp.sum((x-y)**2)
    val = jnp.sum(w*(y-x))*(16*sigma**2)*(2*sigma**2*(2+d)+(-4+d)*dist2)/(2*sigma**2+dist2)**4
    return val

def Delta_x_Delta_y_kappa(x,y,d,sigma):
    dist2 = jnp.sum((x-y)**2)
    val = 16*sigma**2*(4*sigma**4*d*(d+2)+4*sigma**2*(-12-4*d+d**2)*dist2+(24-10*d+d**2)*dist2**2)/(2*sigma**2+dist2)**5
    return val





