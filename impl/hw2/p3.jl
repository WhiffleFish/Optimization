using StaticArrays
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

const SPRING_LOCS = [
    SA[0.0,0.0,0.0],
    SA[1.0,0.0,0.0],
    SA[0.0,0.0,2.0]
]

const UNSTRETCHED_LENGTHS = [
    1., 1., 2.
]

const SPRING_CONSTANTS = [3.0, 12.0, 94.0]

const WEIGHT = 20.0 # N 

function potential(x::AbstractVector)
    spring_potential = 0.5 * mapreduce(+, SPRING_LOCS, SPRING_CONSTANTS, UNSTRETCHED_LENGTHS) do p_i, k_i, l_i
        # k_i * sum(abs2, p_i .- x)
        k_i * (norm(p_i .- x, 2) - l_i)^2
    end
    weight_potential = WEIGHT * x[2]
    return spring_potential + weight_potential
end

f = potential
opt = NesterovMomentum(α=1e-4)
x0 = [0.,-1., 0.]
gf = x -> first(Zygote.gradient(f, x))
x_hist, f_hist = sim(opt, f, gf, x0; max_iter=1000)

p_f = plot(
    f_hist,
    xlabel = "Iteration",
    ylabel = "f(x)"
)

p_g = plot(
    norm.(gf.(x_hist),2), 
    xlabel = "Iteration",
    ylabel = "||∇f(x)||",
    yscale=:log10
)

p = plot(p_f, p_g, lw=2, suptitle="Spring Problem Convergence")

savefig(p, joinpath("figures", "hw2", "p3-convergence.pdf"))

xl = last(x_hist)

norm.(gf.(x_hist),2) |> last

round.(xl, sigdigits=3)
