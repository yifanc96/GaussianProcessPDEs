import jax.numpy as jnp
from jax import grad, jvp, hessian

eps = 0.0

def kappa(x,y,d,sigma):
    dist = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    val = (945*sigma**5+945*jnp.sqrt(11)*sigma**4*dist+4620*sigma**3*dist**2+1155*jnp.sqrt(11)*sigma**2*dist**3+1815*sigma*dist**4+121*jnp.sqrt(11)*dist**5)/(945*sigma**5)*jnp.exp(-jnp.sqrt(11)*dist/sigma)
    return val

def D_wy_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -11*(105*a**4+105*jnp.sqrt(11)*a**3*t+495*a**2*t**2+110*jnp.sqrt(11)*a*t**3+121*t**4)*jnp.exp(-jnp.sqrt(11)*t/a)/(945*a**6)
    val = -DF*sum((x-y)*w)
    return val

def Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -11*(105*d*a**5+105*d*jnp.sqrt(11)*a**4*t+165*(3*d-1)*a**3*t**2+55*jnp.sqrt(11)*(2*d-3)*a**2*t**3+121*(d-6)*a*t**4-121*jnp.sqrt(11)*t**5)/(945*a**7)*jnp.exp(-jnp.sqrt(11)*t/a)
    val = D2F
    return val

def D_wx_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -11*(105*a**4+105*jnp.sqrt(11)*a**3*t+495*a**2*t**2+110*jnp.sqrt(11)*a*t**3+121*t**4)*jnp.exp(-jnp.sqrt(11)*t/a)/(945*a**6)
    val = DF*sum((x-y)*w)
    return val

# Dx vector
def D_x_kappa(x,y,d, sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -11*(105*a**4+105*jnp.sqrt(11)*a**3*t+495*a**2*t**2+110*jnp.sqrt(11)*a*t**3+121*t**4)*jnp.exp(-jnp.sqrt(11)*t/a)/(945*a**6)
    val = DF*(x-y)
    return val

def D_wx_D_wy_kappa(x,y,d,sigma,wx,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -11*(105*a**4+105*jnp.sqrt(11)*a**3*t+495*a**2*t**2+110*jnp.sqrt(11)*a*t**3+121*t**4)*jnp.exp(-jnp.sqrt(11)*t/a)/(945*a**6)
    DDF = 121*jnp.exp(-jnp.sqrt(11)*t/a)*(15*a**3+15*jnp.sqrt(11)*a**2*t+66*a*t**2+11*jnp.sqrt(11)*t**3)/(945*a**7)
    vec = x-y
    val = sum(-wx*wy)*DF+sum(wx*vec)*sum(-wy*vec)*DDF
    return val

# # DxDwy vector
def D_x_D_wy_kappa(x,y,d,sigma,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -11*(105*a**4+105*jnp.sqrt(11)*a**3*t+495*a**2*t**2+110*jnp.sqrt(11)*a*t**3+121*t**4)*jnp.exp(-jnp.sqrt(11)*t/a)/(945*a**6)
    DDF = 121*jnp.exp(-jnp.sqrt(11)*t/a)*(15*a**3+15*jnp.sqrt(11)*a**2*t+66*a*t**2+11*jnp.sqrt(11)*t**3)/(945*a**7)
    vec = x-y
    val = -wy*DF + vec*sum(-wy*vec)*DDF
    return val

def D_wx_Delta_y_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 121*jnp.exp(-jnp.sqrt(11)*t/a)*(15*a**4*(2+d)+15*jnp.sqrt(11)*a**3*(2+d)*t+33*a**2*(3+2*d)*t**2+11*jnp.sqrt(11)*a*(d-1)*t**3-121*t**4)/(945*a**8)
    vec = x-y
    val = D3F*sum(vec*w)
    return val

# # Delta
def Delta_x_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -11*(105*d*a**5+105*d*jnp.sqrt(11)*a**4*t+165*(3*d-1)*a**3*t**2+55*jnp.sqrt(11)*(2*d-3)*a**2*t**3+121*(d-6)*a*t**4-121*jnp.sqrt(11)*t**5)/(945*a**7)*jnp.exp(-jnp.sqrt(11)*t/a)
    
    val = D2F
    return val

def Delta_x_D_wy_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 121*jnp.exp(-jnp.sqrt(11)*t/a)*(15*a**4*(2+d)+15*jnp.sqrt(11)*a**3*(2+d)*t+33*a**2*(3+2*d)*t**2+11*jnp.sqrt(11)*a*(d-1)*t**3-121*t**4)/(945*a**8)
    vec = x-y
    val = -D3F*sum(vec*w)
    return val

def Delta_x_Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D4F = 121*(15*d*(2+d)*a**5+15*d*(2+d)*jnp.sqrt(11)*a**4*t+66*(d**2+d-2)*a**3*t**2+11*jnp.sqrt(11)*a**2*(d**2-4*d-12)*t**3-121*(3+2*d)*a*t**4+121*jnp.sqrt(11)*t**5)/(945*a**9)*jnp.exp(-jnp.sqrt(11)*t/a)
    val = D4F
    return val
