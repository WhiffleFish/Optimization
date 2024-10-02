using LinearAlgebra
using Zygote
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

Base.@kwdef struct NesterovMomentum
    α::Float64 = 1e-2
    γ::Float64 = 0.9
    v::Vector{Float64} = Float64[]
end

function initialize!(m::NesterovMomentum, x)
    resize!(m.v, length(x))
    m.v .= 0
end

function step!(m::NesterovMomentum, f, ∇f, x)
    (; α, γ, v) = m
    x_lookahead = @. x - γ * v
    g = ∇f(x_lookahead)
    v .= γ*v + α*g
    return @. x - v
end

Base.@kwdef struct BackTrackingLineSearch
    α::Float64      = 0.1
    γ::Float64      = 0.9
    β::Float64      = 1e-2
    max_iter::Int   = 100
end

function (ls::BackTrackingLineSearch)(f, x, d)
    (;α, γ) = ls
    fx = f(x)
    xp = x + α * d
    fxp = f(xp)
    γt = γ
    t = 1
    while fxp > fx
        @. xp = x + γt * α * d
        fxp = f(xp)
        γt *= γ
        t += 1
    end
    return xp
end

# armijo
function (ls::BackTrackingLineSearch)(f, ∇f, x, d)
    (;α, γ, β) = ls
    fx = f(x)
    gd = dot(∇f(x), d)
    xp = copy(x)
    while f(@. xp = x + α * d) > fx + β * α * gd
        α *= γ
    end
    # @show fx, f(xp), α, fx + β * α * gd
    return xp
end

Base.@kwdef mutable struct BFGS{LS}
    Q::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
    line_search::LS = BackTrackingLineSearch(α = 1e-1)
end

function initialize!(m::BFGS, x)
    m.Q = convert(Matrix{Float64}, I(length(x)))
end

function step!(m::BFGS, f, ∇f, x)
    (; Q, line_search) = m
    g = ∇f(x)
    xp = line_search(f, ∇f, x, -Q*g)
    gp = ∇f(xp)
    γ = gp - g
    δ = xp - x
    m.Q .= Q - 
        (δ * γ' * Q + Q * γ * δ') / dot(δ, γ) + 
        (1 + dot(γ, Q, γ)/dot(δ, γ)) * (δ * δ') / dot(δ, γ)
    return xp
end

function sim(opt, f, ∇f, x; max_iter=100, stop_crit = Returns(false))
    x_hist = [x]
    f_hist = [f(x)]
    initialize!(opt, x)
    for i ∈ 1:max_iter
        x = step!(opt, f, ∇f, x)
        push!(x_hist, x)
        push!(f_hist, f(x))
    end
    return x_hist, f_hist
end

rosenbrock(x; a=1, b=5) = (a-x[1])^2 + b*(x[2] - x[1]^2)^2

X = Y = range(-2, stop=2, length=100)
contour(X, Y, (a,b)->rosenbrock((a,b)), levels=100)

x0 = [-1.2, 1]
f = rosenbrock
∇f(x) = first(Zygote.gradient(f, x))

opt = NesterovMomentum()
x_nest, f_nest = sim(opt, f, ∇f, x0, max_iter=1000)
g_nest = norm.(∇f.(x_nest), 2)

opt = BFGS(line_search = BackTrackingLineSearch(α=1e-1, β=1e-2))
x_bfgs, f_bfgs = sim(opt, f, ∇f, x0, max_iter=1000)
g_bfgs = norm.(∇f.(x_bfgs), 2)


X = Y = range(-2, stop=2, length=100)
p = contour(X, Y, (a,b)->rosenbrock((a,b)), levels=100)
plot!(p, first.(x_nest), last.(x_nest), label="Nesterov")
plot!(p, first.(x_bfgs), last.(x_bfgs), label="BFGS")
savefig(p, joinpath("figures", "hw2", "p2-contour.pdf"))

bfgs_conv = findfirst(≤(1e-4), g_bfgs)
nest_conv = findfirst(≤(1e-4), g_nest)

p_f = plot(f_nest, label="Nesterov", xlabel="Iteration", ylabel="f(x)")
plot!(p_f, f_bfgs, label="BFGS")
p_g = plot(
    g_nest .+ eps(), 
    yscale=:log10,  
    xlabel="Iteration", 
    ylabel= "||∇f(x)||", 
    label="Nesterov", 
    legend=:topright
)
plot!(p_g, g_bfgs .+ eps(), yscale=:log10, label="BFGS")
vline!(p_g, [nest_conv bfgs_conv], 
    labels = ["Nesterov Convergence" "BFGS Convergence"]
)

p = plot(p_f, p_g)

savefig(p, joinpath("figures", "hw2", "p2-convergence.pdf"))


