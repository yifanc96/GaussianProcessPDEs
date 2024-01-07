import jax.numpy as jnp

def kappa(x1, x2, y1, y2, sigma):
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    return jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))

# cover derivatives over x1 and y1

def D_wy1_kappa(x1, x2, y1, y2, sigma, w):
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = -sum(w*(y1-x1))/(sigma[0]**2)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val



def Delta_y1_kappa(x1, x2, y1, y2, sigma):
    d = jnp.shape(x1)[0]
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = (-d*(sigma[0]**2)+sdist1)/(sigma[0]**4)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

def D_wx1_kappa(x1, x2, y1, y2, sigma,w):
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = -sum(w*(x1-y1))/(sigma[0]**2)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

# Dx vector
def D_x1_kappa(x1, x2, y1, y2, sigma):
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = -(x1-y1)/(sigma[0]**2)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

def D_wx1_D_wy1_kappa(x1, x2, y1, y2, sigma,wx1,wy1):
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = (sum(wx1*wy1)/(sigma[0]**2)+sum(wx1*(x1-y1))*sum(wy1*(y1-x1))/(sigma[0]**4))*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

# # DxDwy1 vector
def D_x1_D_wy1_kappa(x1, x2, y1, y2, sigma,wy1):
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = (wy1/(sigma[0]**2)+(x1-y1)*sum(wy1*(y1-x1))/(sigma[0]**4))*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

def D_wx1_Delta_y1_kappa(x1, x2, y1, y2, sigma,w):
    d = jnp.shape(x1)[0]
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = sum(w*(x1-y1))*((sigma[0]**2)*(2+d)-sdist1)/(sigma[0]**6)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

# # Delta
def Delta_x1_kappa(x1, x2, y1, y2, sigma):
    d = jnp.shape(x1)[0]
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = (-d*(sigma[0]**2)+sdist1)/(sigma[0]**4)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

def Delta_x1_D_wy1_kappa(x1, x2, y1, y2, sigma,w):
    d = jnp.shape(x1)[0]
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = sum(w*(y1-x1))*((sigma[0]**2)*(2+d)-sdist1)/(sigma[0]**6)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

def Delta_x1_Delta_y1_kappa(x1, x2, y1, y2, sigma):
    d = jnp.shape(x1)[0]
    sdist1 = sum((x1-y1)**2)
    sdist2 = sum((x2-y2)**2)
    val = ((sigma[0]**4)*d*(2+d)-2*(sigma[0]**2)*(2+d)*sdist1+sdist1**2)/(sigma[0]**8)*jnp.exp(-sdist1/(2*sigma[0]**2) - sdist2/(2*sigma[1]**2))
    return val

