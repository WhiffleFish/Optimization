using Distributions
using StaticArrays
using StatsBase
using LinearAlgebra
using Optimization
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

camel(x,y) = (4 - 2.1x^2 + (x^4)/3) * x^2 + x*y + (-4 + 4y^2)*y^2 + 0.1y

camel(x::AbstractArray) = camel(first(x), last(x))

opt = CrossEntropy(ϵ=-Inf, max_iter=100)
d0 = MvNormal([0.0, 0.0], diagm([3.0, 3.0]))
f_ce, x_ce, hist_ce = optimize_info(opt, camel, d0)
plot(hist_ce.f)

opt = HookeJeeves(ϵ=-Inf,γ=0.9, max_iter=100)
x0 = [0.0, 0.0]
f_hj, x_hj, hist_hj = optimize_info(opt, camel, x0)
plot(hist_hj.f)

p = plot(hist_ce.f, lw=2, xlabel="Iteration", ylabel="Best iterate value", title="Six-Hump Camel Optimization", label="Cross Entropy")
plot!(p, hist_hj.f, lw=2, label="Hooke Jeeves")
savefig(p, joinpath("figures", "hw3", "p1-CE-HJ-comparison.pdf"))
