using Distributions
using StaticArrays
using StatsBase
using LinearAlgebra
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

camel(x,y) = (4 - 2.1x^2 + (x^4)/3) * x^2 + x*y + (-4 + 4y^2)*y^2 + 0.1y
camel(x::AbstractArray) = camel(first(x), last(x))
camel_g(x,y) = [x^3 - y]
camel_g(x::AbstractArray) = camel_g(first(x), last(x))

camel_obj_and_constraints(v) = camel(v), camel_g(v)

opt = PenaltyOptimizer(HookeJeeves(max_iter=1000, ϵ=-Inf), ρ=0.1, γ=1.003, max_iter=1000)
f,g,x,hist = optimize_info(opt, camel_obj_and_constraints, [0.0, 0.0])


Xs = range(-2.0, 2.0, length=100)
Ys = range(-1.0, 1.0, length=100)

plot(hist.f)


contourf(Xs, Ys, 
    (x,y) -> camel(x,y),
    xlims = extrema(Xs),
    ylims = extrema(Ys)
)
plot!(Xs, x-> x^3, lw=5, c=:green)
plot!(first.(hist.x), last.(hist.x), c=:blue, ms=10, markershape=:star)

plot(last.(hist.x), c=:blue)
