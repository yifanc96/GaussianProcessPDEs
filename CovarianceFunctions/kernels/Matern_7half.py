import jax.numpy as jnp

eps = 0.0

def kappa(x,y,d,sigma):
    dist = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    val = (15*sigma**3+15*jnp.sqrt(7)*(sigma**2)*dist+42*sigma*(dist**2)+7*jnp.sqrt(7)*dist**3)/(15*sigma**3)*jnp.exp(-jnp.sqrt(7)*dist/sigma)
    return val

def D_wy_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -7*(3*a**2+3*jnp.sqrt(7)*a*t+7*t**2)*jnp.exp(-jnp.sqrt(7)*t/a)/(15*a**4)
    val = -DF*sum((x-y)*w)
    return val

def Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -7*(3*d*a**3+3*jnp.sqrt(7)*a**2*d*t+7*a*(d-1)*t**2-7*jnp.sqrt(7)*t**3)/(15*a**5)*jnp.exp(-jnp.sqrt(7)*t/a)
    val = D2F
    return val

def D_wx_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -7*(3*a**2+3*jnp.sqrt(7)*a*t+7*t**2)*jnp.exp(-jnp.sqrt(7)*t/a)/(15*a**4)
    val = DF*sum((x-y)*w)
    return val

# Dx vector
def D_x_kappa(x,y,d, sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -7*(3*a**2+3*jnp.sqrt(7)*a*t+7*t**2)*jnp.exp(-jnp.sqrt(7)*t/a)/(15*a**4)
    val = DF*(x-y)
    return val

def D_wx_D_wy_kappa(x,y,d,sigma,wx,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -7*(3*a**2+3*jnp.sqrt(7)*a*t+7*t**2)*jnp.exp(-jnp.sqrt(7)*t/a)/(15*a**4)
    DDF = 49*jnp.exp(-jnp.sqrt(7)*t/a)*(a+jnp.sqrt(7)*t)/(15*a**5)
    vec = x-y
    val = sum(-wx*wy)*DF+sum(wx*vec)*sum(-wy*vec)*DDF
    return val

# # DxDwy vector
def D_x_D_wy_kappa(x,y,d,sigma,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -7*(3*a**2+3*jnp.sqrt(7)*a*t+7*t**2)*jnp.exp(-jnp.sqrt(7)*t/a)/(15*a**4)
    DDF = 49*jnp.exp(-jnp.sqrt(7)*t/a)*(a+jnp.sqrt(7)*t)/(15*a**5)
    vec = x-y
    val = -wy*DF + vec*sum(-wy*vec)*DDF
    return val

def D_wx_Delta_y_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 49*jnp.exp(-jnp.sqrt(7)*t/a)*(a**2*(2+d)+jnp.sqrt(7)*a*(2+d)*t-7*t**2)/(15*a**6)
    vec = x-y
    val = D3F*sum(vec*w)
    return val

# # Delta
def Delta_x_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -7*(3*d*a**3+3*jnp.sqrt(7)*a**2*d*t+7*a*(d-1)*t**2-7*jnp.sqrt(7)*t**3)/(15*a**5)*jnp.exp(-jnp.sqrt(7)*t/a)
    
    val = D2F
    return val

def Delta_x_D_wy_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 49*jnp.exp(-jnp.sqrt(7)*t/a)*(a**2*(2+d)+jnp.sqrt(7)*a*(2+d)*t-7*t**2)/(15*a**6)
    vec = x-y
    val = -D3F*sum(vec*w)
    return val

def Delta_x_Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D4F = 49*(d*(d+2)*a**3+d*(d+2)*jnp.sqrt(7)*a**2*t-14*a*(2+d)*t**2+7*jnp.sqrt(7)*t**3)/(15*a**7)*jnp.exp(-jnp.sqrt(7)*t/a)
    val = D4F
    return val
