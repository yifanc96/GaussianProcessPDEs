include("./Measurements.jl")

abstract type AbstractCovarianceFunction{Tv} end 

# automatically implements a mutating batched version of a given covariance function 
function (cov::AbstractCovarianceFunction{Tv})(out::AbstractMatrix{Tv}, x_vec::AbstractVector{<:AbstractMeasurement}, y_vec::AbstractVector{<:AbstractMeasurement}) where Tv
    for cartesian in CartesianIndices(out) 
        out[cartesian] = cov(x_vec[cartesian[1]], y_vec[cartesian[2]])
    end
end

# automatically implements a mutating batched version of a given covariance function, using symmetry
function (cov::AbstractCovarianceFunction{Tv})(out::AbstractMatrix{Tv}, x_vec::AbstractVector{<:AbstractMeasurement}) where Tv
    for cartesian in CartesianIndices(out) 
        if cartesian[1] >= cartesian[2]
            out[cartesian] = cov(x_vec[cartesian[1]], x_vec[cartesian[2]])
        else
            out[cartesian] = out[cartesian[2], cartesian[1]]
        end
    end
end

struct MaternCovariance1_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end
function (cov::MaternCovariance1_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return exp(-dist/sigma)
end

struct MaternCovariance3_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end
function (cov::MaternCovariance3_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return (1+sqrt(3)*dist/sigma)*exp(-sqrt(3)*dist/sigma)
end

struct MaternCovariance5_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

function (cov::MaternCovariance5_2)(x::PointMeasurement, y::PointMeasurement)
    # d = length(x.coordinate)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    F(t,a) = (1 + sqrt(5)*t/a + 5*t^2/(3*a^2))*exp(-sqrt(5)*t/a)
    return F(dist,sigma)
end

function (cov::MaternCovariance5_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;

    F(t,a) = (1 + sqrt(5)*t/a + 5*t^2/(3*a^2))*exp(-sqrt(5)*t/a)
    D2F(t,a) = -5*(d*a^2+sqrt(5)*d*a*t-5*t^2)/(3*a^4) * exp(-sqrt(5)*t/a);
    D4F(t,a) = 25*(d*(d+2)*a^2-(3+2*d)*sqrt(5)*a*t+5*t^2)/(3*a^6) * exp(-sqrt(5)*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end

function (cov::MaternCovariance5_2)(x::Δ∇δPointMeasurement, y::Δ∇δPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    wg_x = x.weight_∇
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    wg_y = y.weight_∇

    F(t,a) = (1 + sqrt(5)*t/a + 5*t^2/(3*a^2))*exp(-sqrt(5)*t/a)
    D2F(t,a) = -5*(d*a^2+sqrt(5)*d*a*t-5*t^2)/(3*a^4) * exp(-sqrt(5)*t/a)
    D4F(t,a) = 25*(d*(d+2)*a^2-(3+2*d)*sqrt(5)*a*t+5*t^2)/(3*a^6) * exp(-sqrt(5)*t/a)
    DF(t,a) = -5*(a+sqrt(5)*t)*exp(-sqrt(5)*t/a)/(3*a^3)
    D3F(t,a) = 25*exp(-sqrt(5)*t/a)*(a*(2+d)-sqrt(5)*t)/(3*a^5)
    DDF(t,a) = 25*exp(-sqrt(5)*t/a)/(3*a^4)
    vec = x.coordinate - y.coordinate
    dist = norm(vec);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma) - w1_x*D3F(dist,sigma)*sum(vec.*wg_y) + w1_y*D3F(dist,sigma)*sum(vec.*wg_x) - w2_x*DF(dist,sigma)*sum(vec.*wg_y) + w2_y*DF(dist,sigma)*sum(vec.*wg_x) + (sum(-wg_x.*wg_y)*DF(dist,sigma)+sum(wg_x.*vec)*sum(-wg_y.*vec)*DDF(dist,sigma))
end

struct MaternCovariance7_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

function (cov::MaternCovariance7_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    F(t,a) = (15*a^3+15*sqrt(7)*a^2*t+42*a*t^2+7*sqrt(7)*t^3)/(15*a^3)*exp(-sqrt(7)*t/a);
    return F(dist,sigma)
end

function (cov::MaternCovariance7_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    F(t,a) = (15*a^3+15*sqrt(7)*a^2*t+42*a*t^2+7*sqrt(7)*t^3)/(15*a^3)*exp(-sqrt(7)*t/a);
    D2F(t,a) = -7*(3*d*a^3+3*sqrt(7)*a^2*d*t+7*a*(d-1)*t^2-7*sqrt(7)*t^3)/(15*a^5)*exp(-sqrt(7)*t/a);
    D4F(t,a) = 49*(d*(d+2)*a^3+d*(d+2)*sqrt(7)*a^2*t-14*a*(2+d)*t^2+7*sqrt(7)*t^3)/(15*a^7)*exp(-sqrt(7)*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end

function (cov::MaternCovariance7_2)(x::Δ∇δPointMeasurement, y::Δ∇δPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    wg_x = x.weight_∇
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    wg_y = y.weight_∇

    F(t,a) = (15*a^3+15*sqrt(7)*a^2*t+42*a*t^2+7*sqrt(7)*t^3)/(15*a^3)*exp(-sqrt(7)*t/a);
    D2F(t,a) = -7*(3*d*a^3+3*sqrt(7)*a^2*d*t+7*a*(d-1)*t^2-7*sqrt(7)*t^3)/(15*a^5)*exp(-sqrt(7)*t/a);
    D4F(t,a) = 49*(d*(d+2)*a^3+d*(d+2)*sqrt(7)*a^2*t-14*a*(2+d)*t^2+7*sqrt(7)*t^3)/(15*a^7)*exp(-sqrt(7)*t/a);
    DF(t,a) = -7*(3*a^2+3*sqrt(7)*a*t+7*t^2)*exp(-sqrt(7)*t/a)/(15*a^4)
    D3F(t,a) = 49*exp(-sqrt(7)*t/a)*(a^2*(2+d)+sqrt(7)*a*(2+d)*t-7*t^2)/(15*a^6)
    DDF(t,a) = 49*exp(-sqrt(7)*t/a)*(a+sqrt(7)*t)/(15*a^5)
    vec = x.coordinate - y.coordinate
    dist = norm(vec);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma) - w1_x*D3F(dist,sigma)*sum(vec.*wg_y) + w1_y*D3F(dist,sigma)*sum(vec.*wg_x) - w2_x*DF(dist,sigma)*sum(vec.*wg_y) + w2_y*DF(dist,sigma)*sum(vec.*wg_x) + (sum(-wg_x.*wg_y)*DF(dist,sigma)+sum(wg_x.*vec)*sum(-wg_y.*vec)*DDF(dist,sigma))
end


struct MaternCovariance9_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

# Matern covariance function
function (cov::MaternCovariance9_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    F(t,a) = (35*a^4+105*a^3*t+135*a^2*t^2+90*a*t^3+27*t^4)/(35*a^4)*exp(-3*t/a);
    return F(dist,sigma)
end

function (cov::MaternCovariance9_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    F(t,a) = (35*a^4+105*a^3*t+135*a^2*t^2+90*a*t^3+27*t^4)/(35*a^4)*exp(-3*t/a);
    D2F(t,a) = -9*(5*d*a^4+15*d*a^3*t+9*a^2*(2*d-1)*t^2+9*a*(d-3)*t^3-27*t^4)/(35*a^6)*exp(-3*t/a);
    D4F(t,a) = 81*(d*(d+2)*a^4+3*a^3*d*(d+2)*t+3*a^2*(d^2-4)-18*a*(d+2)*t^3+27*t^4)/(35*a^8)*exp(-3*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end

function (cov::MaternCovariance9_2)(x::Δ∇δPointMeasurement, y::Δ∇δPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    wg_x = x.weight_∇
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    wg_y = y.weight_∇

    F(t,a) = (35*a^4+105*a^3*t+135*a^2*t^2+90*a*t^3+27*t^4)/(35*a^4)*exp(-3*t/a);
    D2F(t,a) = -9*(5*d*a^4+15*d*a^3*t+9*a^2*(2*d-1)*t^2+9*a*(d-3)*t^3-27*t^4)/(35*a^6)*exp(-3*t/a);
    D4F(t,a) = 81*(d*(d+2)*a^4+3*a^3*d*(d+2)*t+3*a^2*(d^2-4)*(t^2)-18*a*(d+2)*t^3+27*t^4)/(35*a^8)*exp(-3*t/a);
    DF(t,a) = -9*(5*a^3+15*a^2*t+18*a*t^2+9*t^3)*exp(-3*t/a)/(35*a^5)
    D3F(t,a) = 81*exp(-3*t/a)*(a^3*(2+d)+3*a^2*(2+d)*t+3*a*(1+d)*t^2-9*t^3)/(35*a^7)
    DDF(t,a) = 81*exp(-3*t/a)*(a^2+3*a*t+3*t^2)/(35*a^6)
    vec = x.coordinate - y.coordinate
    dist = norm(vec);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma) - w1_x*D3F(dist,sigma)*sum(vec.*wg_y) + w1_y*D3F(dist,sigma)*sum(vec.*wg_x) - w2_x*DF(dist,sigma)*sum(vec.*wg_y) + w2_y*DF(dist,sigma)*sum(vec.*wg_x) + (sum(-wg_x.*wg_y)*DF(dist,sigma)+sum(wg_x.*vec)*sum(-wg_y.*vec)*DDF(dist,sigma))
end


struct MaternCovariance11_2{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

# Matern covariance function
function (cov::MaternCovariance11_2)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    F(t,a) = (945*a^5+945*sqrt(11)*a^4*t+4620*a^3*t^2+1155*sqrt(11)*a^2*t^3+1815*a*t^4+121*sqrt(11)*t^5)/(945*a^5)*exp(-sqrt(11)*t/a);
    return F(dist,sigma)
end

function (cov::MaternCovariance11_2)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    F(t,a) = (945*a^5+945*sqrt(11)*a^4*t+4620*a^3*t^2+1155*sqrt(11)*a^2*t^3+1815*a*t^4+121*sqrt(11)*t^5)/(945*a^5)*exp(-sqrt(11)*t/a);
    D2F(t,a) = -11*(105*d*a^5+105*d*sqrt(11)*a^4*t+165*(3*d-1)*a^3*t^2+55*sqrt(11)*(2*d-3)*a^2*t^3+121*(d-6)*a*t^4-121*sqrt(11)*t^5)/(945*a^7)*exp(-sqrt(11)*t/a);
    D4F(t,a) = 121*(15*d*(2+d)*a^5+15*d*(2+d)*sqrt(11)*a^4*t+66*(d^2+d-2)*a^3*t^2+11*sqrt(11)*a^2*(d^2-4*d-12)*t^3-121*(3+2*d)*a*t^4+121*sqrt(11)*t^5)/(945*a^9)*exp(-sqrt(11)*t/a);
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end

function (cov::MaternCovariance11_2)(x::Δ∇δPointMeasurement, y::Δ∇δPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    wg_x = x.weight_∇
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    wg_y = y.weight_∇

    F(t,a) = (945*a^5+945*sqrt(11)*a^4*t+4620*a^3*t^2+1155*sqrt(11)*a^2*t^3+1815*a*t^4+121*sqrt(11)*t^5)/(945*a^5)*exp(-sqrt(11)*t/a);
    D2F(t,a) = -11*(105*d*a^5+105*d*sqrt(11)*a^4*t+165*(3*d-1)*a^3*t^2+55*sqrt(11)*(2*d-3)*a^2*t^3+121*(d-6)*a*t^4-121*sqrt(11)*t^5)/(945*a^7)*exp(-sqrt(11)*t/a);
    D4F(t,a) = 121*(15*d*(2+d)*a^5+15*d*(2+d)*sqrt(11)*a^4*t+66*(d^2+d-2)*a^3*t^2+11*sqrt(11)*a^2*(d^2-4*d-12)*t^3-121*(3+2*d)*a*t^4+121*sqrt(11)*t^5)/(945*a^9)*exp(-sqrt(11)*t/a);
    DF(t,a) = -11*(105*a^4+105*sqrt(11)*a^3*t+495*a^2*t^2+110*sqrt(11)*a*t^3+121*t^4)*exp(-sqrt(11)*t/a)/(945*a^6)
    D3F(t,a) = 121*exp(-sqrt(11)*t/a)*(15*a^4*(2+d)+15*sqrt(11)*a^3*(2+d)*t+33*a^2*(3+2*d)*t^2+11*sqrt*(11)*a*(d-1)*t^3-121*t^4)/(945*a^8)
    DDF(t,a) = 121*exp(-sqrt(11)*t/a)*(15*a^3+15*sqrt(11)*a^2*t+66*a*t^2+11*sqrt(11)*t^3)/(945*a^7)
    vec = x.coordinate - y.coordinate
    dist = norm(vec);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma) - w1_x*D3F(dist,sigma)*sum(vec.*wg_y) + w1_y*D3F(dist,sigma)*sum(vec.*wg_x) - w2_x*DF(dist,sigma)*sum(vec.*wg_y) + w2_y*DF(dist,sigma)*sum(vec.*wg_x) + (sum(-wg_x.*wg_y)*DF(dist,sigma)+sum(wg_x.*vec)*sum(-wg_y.*vec)*DDF(dist,sigma))
end


# The Gaussian covariance function
struct GaussianCovariance{Tv}<:AbstractCovarianceFunction{Tv}
    length_scale::Tv
end

function (cov::GaussianCovariance)(x::PointMeasurement, y::PointMeasurement)
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return exp(-dist^2/(2*sigma^2))
end

function (cov::GaussianCovariance)(x::ΔδPointMeasurement, y::ΔδPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    F(t,a) = exp(-t^2/(2*a^2))
    D2F(t,a) = (t^2 - a^2*d)/(a^4)*exp(-t^2/(2*a^2));
    D4F(t,a) = (a^4*d*(2+d)-2*a^2*(2+d)*t^2+t^4)*exp(-t^2/(2*a^2))/a^8
    dist = norm(x.coordinate - y.coordinate);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma)
end

function (cov::GaussianCovariance)(x::Δ∇δPointMeasurement, y::Δ∇δPointMeasurement)
    d = length(x.coordinate);
    w1_x = x.weight_Δ;
    w2_x = x.weight_δ;
    wg_x = x.weight_∇
    w1_y = y.weight_Δ;
    w2_y = y.weight_δ;
    wg_y = y.weight_∇

    F(t,a) = exp(-t^2/(2*a^2))
    D2F(t,a) = (t^2 - a^2*d)/(a^4)*exp(-t^2/(2*a^2));
    D4F(t,a) = (a^4*d*(2+d)-2*a^2*(2+d)*t^2+t^4)*exp(-t^2/(2*a^2))/a^8
    DF(t,a) = -exp(-t^2/(2*a^2))/a^2
    D3F(t,a) = exp(-t^2/(2*a^2))*(a^2*(2+d)*-t^2)/a^6
    DDF(t,a) = exp(-t^2/(2*a^2))/a^4
    vec = x.coordinate - y.coordinate
    dist = norm(vec);
    sigma = cov.length_scale;
    return w1_x*w1_y*D4F(dist,sigma) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,sigma) + w2_x*w2_y*F(dist,sigma) - w1_x*D3F(dist,sigma)*sum(vec.*wg_y) + w1_y*D3F(dist,sigma)*sum(vec.*wg_x) - w2_x*DF(dist,sigma)*sum(vec.*wg_y) + w2_y*DF(dist,sigma)*sum(vec.*wg_x) + (sum(-wg_x.*wg_y)*DF(dist,sigma)+sum(wg_x.*vec)*sum(-wg_y.*vec)*DDF(dist,sigma))
end


function (cov::AbstractCovarianceFunction)(x::ΔδPointMeasurement, y::PointMeasurement)
    return cov(x, ΔδPointMeasurement(y))
end
function (cov::AbstractCovarianceFunction)(x::PointMeasurement, y::ΔδPointMeasurement)
    return (cov::AbstractCovarianceFunction)(y,x)
end

function (cov::AbstractCovarianceFunction)(x::Δ∇δPointMeasurement, y::PointMeasurement)
    return cov(x, Δ∇δPointMeasurement(y))
end
function (cov::AbstractCovarianceFunction)(x::PointMeasurement, y::Δ∇δPointMeasurement)
    return (cov::AbstractCovarianceFunction)(y,x)
end
