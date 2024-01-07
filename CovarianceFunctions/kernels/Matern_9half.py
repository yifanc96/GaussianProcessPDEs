import jax.numpy as jnp

eps = 0.0

def kappa(x,y,d,sigma):
    dist = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    val = (35*sigma**4+105*(sigma**3)*dist+135*sigma**2*dist**2+90*sigma*dist**3+27*dist**4)/(35*sigma**4)*jnp.exp(-3*dist/sigma)
    return val

def D_wy_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -9*(5*a**3+15*a**2*t+18*a*t**2+9*t**3)*jnp.exp(-3*t/a)/(35*a**5)
    val = -DF*sum((x-y)*w)
    return val

def Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -9*(5*d*a**4+15*d*a**3*t+9*a**2*(2*d-1)*t**2+9*a*(d-3)*t**3-27*t**4)/(35*a**6)*jnp.exp(-3*t/a)
    val = D2F
    return val

def D_wx_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -9*(5*a**3+15*a**2*t+18*a*t**2+9*t**3)*jnp.exp(-3*t/a)/(35*a**5)
    val = DF*sum((x-y)*w)
    return val

# Dx vector
def D_x_kappa(x,y,d, sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -9*(5*a**3+15*a**2*t+18*a*t**2+9*t**3)*jnp.exp(-3*t/a)/(35*a**5)
    val = DF*(x-y)
    return val

def D_wx_D_wy_kappa(x,y,d,sigma,wx,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -9*(5*a**3+15*a**2*t+18*a*t**2+9*t**3)*jnp.exp(-3*t/a)/(35*a**5)
    DDF = 81*jnp.exp(-3*t/a)*(a**2+3*a*t+3*t**2)/(35*a**6)
    vec = x-y
    val = sum(-wx*wy)*DF+sum(wx*vec)*sum(-wy*vec)*DDF
    return val

# # DxDwy vector
def D_x_D_wy_kappa(x,y,d,sigma,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -9*(5*a**3+15*a**2*t+18*a*t**2+9*t**3)*jnp.exp(-3*t/a)/(35*a**5)
    DDF = 81*jnp.exp(-3*t/a)*(a**2+3*a*t+3*t**2)/(35*a**6)
    vec = x-y
    val = -wy*DF + vec*sum(-wy*vec)*DDF
    return val

def D_wx_Delta_y_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 81*jnp.exp(-3*t/a)*(a**3*(2+d)+3*a**2*(2+d)*t+3*a*(1+d)*t**2-9*t**3)/(35*a**7)
    vec = x-y
    val = D3F*sum(vec*w)
    return val

# # Delta
def Delta_x_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -9*(5*d*a**4+15*d*a**3*t+9*a**2*(2*d-1)*t**2+9*a*(d-3)*t**3-27*t**4)/(35*a**6)*jnp.exp(-3*t/a)
    
    val = D2F
    return val

def Delta_x_D_wy_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 81*jnp.exp(-3*t/a)*(a**3*(2+d)+3*a**2*(2+d)*t+3*a*(1+d)*t**2-9*t**3)/(35*a**7)
    vec = x-y
    val = -D3F*sum(vec*w)
    return val

def Delta_x_Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D4F = 81*(d*(d+2)*a**4+3*a**3*d*(d+2)*t+3*a**2*(d**2-4)*t**2-18*a*(d+2)*t**3+27*t**4)/(35*a**8)*jnp.exp(-3*t/a)
    val = D4F
    return val




