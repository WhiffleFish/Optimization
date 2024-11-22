using Zygote
using LinearAlgebra

traj(u) = accumulate(+, map(uk -> 6sin(uk), u))    
full_traj(u) = [0;traj(u)]
car_objective(y) = dot(y,y)
g_control(u) = [u .- 0.1; -u .- 0.1]
_y_l = [fill(-2.0, 4); fill(1.0, 6)]
_y_u = fill(2.0, 10)
g_traj(y) = [y .- _y_u;  -y .+ _y_l]

function car_objective_constraints(u)
    y = traj(u)
    return car_objective(y), vcat(g_traj(y), g_control(u))
end

u = zeros(10)
u[2] = 0.1
y = traj(u)
g_traj(y)
car_objective_constraints(u)

p_quadratic(gs) = sum(gs) do g_i
    max(g_i, 0)^2
end

function full_objective(ρ)
    return function (u)
        J, g = car_objective_constraints(u)
        return 1*J + ρ * p_quadratic(g)
    end
end

f = full_objective(1.0)
f(u)

first(gradient(f, u))


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

function penalty_sim(opt, x; ρ=0.1, γ=1.1, k_max=100, kwargs...)
    f = full_objective(ρ)
    x_hist_hist = []
    f_hist_hist = []
    for i ∈ 1:k_max
        f = full_objective(ρ)
        x_hist, f_hist = sim(opt, f, u->first(gradient(f, u)), x; kwargs...)
        J,g = car_objective_constraints(last(x_hist))
        x = last(x_hist)
        @show J
        @show maximum(g)
        if all(≤(0), g)
            break
        end
        ρ *= γ
        x = last(x_hist)
        push!(x_hist_hist, x_hist)
        push!(f_hist_hist, f_hist)
    end
    return x_hist_hist, f_hist_hist
end

opt = NesterovMomentum(α=1e-7)
x_hist_hist, f_hist_hist = penalty_sim(opt, zeros(10), k_max=100, max_iter=5000)
plot(traj(last(x_hist_hist[100])))
plot(last(x_hist_hist[100]))

last(last(x_hist_hist))

plot(f_hist_hist[100])
plot(traj(last(x_hist_hist[100])))
