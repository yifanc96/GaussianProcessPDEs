import jax.numpy as jnp

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

def Delta_x_Delta_y_kappa(x,y,d, sigma):
    dist2 = sum((x-y)**2)
    val = ((sigma**4)*d*(2+d)-2*(sigma**2)*(2+d)*dist2+dist2**2)/(sigma**8)*jnp.exp(-dist2/(2*sigma**2))
    return val