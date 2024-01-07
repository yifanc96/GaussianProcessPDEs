import jax.numpy as jnp

eps = 0.0

def kappa(x,y,d,sigma):
    dist = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    val = (1+jnp.sqrt(5)*dist/sigma+5*dist**2/(3*sigma**2)) * jnp.exp(-jnp.sqrt(5)*dist/sigma)
    return val

def D_wy_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    
    DF = -5*(a+jnp.sqrt(5)*t)*jnp.exp(-jnp.sqrt(5)*t/a)/(3*a**3)
    val = -DF*sum((x-y)*w)
    return val

def Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -5*(d*a**2+jnp.sqrt(5)*d*a*t-5*t**2)/(3*a**4) * jnp.exp(-jnp.sqrt(5)*t/a)
    val = D2F
    return val

def D_wx_kappa(x,y,d, sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -5*(a+jnp.sqrt(5)*t)*jnp.exp(-jnp.sqrt(5)*t/a)/(3*a**3)
    val = DF*sum((x-y)*w)
    return val

# Dx vector
def D_x_kappa(x,y,d, sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -5*(a+jnp.sqrt(5)*t)*jnp.exp(-jnp.sqrt(5)*t/a)/(3*a**3)
    val = DF*(x-y)
    return val

def D_wx_D_wy_kappa(x,y,d,sigma,wx,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -5*(a+jnp.sqrt(5)*t)*jnp.exp(-jnp.sqrt(5)*t/a)/(3*a**3)
    DDF = 25*jnp.exp(-jnp.sqrt(5)*t/a)/(3*a**4)
    vec = x-y
    val = sum(-wx*wy)*DF+sum(wx*vec)*sum(-wy*vec)*DDF
    return val

# # DxDwy vector
def D_x_D_wy_kappa(x,y,d,sigma,wy):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    DF = -5*(a+jnp.sqrt(5)*t)*jnp.exp(-jnp.sqrt(5)*t/a)/(3*a**3)
    DDF = 25*jnp.exp(-jnp.sqrt(5)*t/a)/(3*a**4)
    vec = x-y
    val = -wy*DF + vec*sum(-wy*vec)*DDF
    return val

def D_wx_Delta_y_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 25*jnp.exp(-jnp.sqrt(5)*t/a)*(a*(2+d)-jnp.sqrt(5)*t)/(3*a**5)
    vec = x-y
    val = D3F*sum(vec*w)
    return val

# # Delta
def Delta_x_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D2F = -5*(d*a**2+jnp.sqrt(5)*d*a*t-5*t**2)/(3*a**4) * jnp.exp(-jnp.sqrt(5)*t/a)
    
    val = D2F
    return val

def Delta_x_D_wy_kappa(x,y, d,sigma,w):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D3F = 25*jnp.exp(-jnp.sqrt(5)*t/a)*(a*(2+d)-jnp.sqrt(5)*t)/(3*a**5)
    vec = x-y
    val = -D3F*sum(vec*w)
    return val

def Delta_x_Delta_y_kappa(x,y,d,sigma):
    t = jnp.sqrt(jnp.sum((x-y)**2 + eps))
    a = sigma
    D4F = 25*(d*(d+2)*a**2-(3+2*d)*jnp.sqrt(5)*a*t+5*t**2)/(3*a**6) * jnp.exp(-jnp.sqrt(5)*t/a)
    val = D4F
    return val



# test
# x = jnp.array([0.0,0.0])
# y = jnp.array([0.0,0.0])
# w = jnp.array([1.0,1.0])
# d = 2
# sigma = 0.2
# print(D_wx_D_wy_kappa(x,y,d,sigma,w,w))




