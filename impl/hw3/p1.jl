using Distributions
using StaticArrays
using StatsBase
using LinearAlgebra
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

camel(x,y) = (4 - 2.1x^2 + (x^4)/3) * x^2 + x*y + (-4 + 4y^2)*y^2 + 0.1y

camel(x::AbstractArray) = camel(first(x), last(x))

struct CrossEntropyOptimization
    n_elite::Int
    n::Int
end

function opt_iter(opt::CrossEntropyOptimization, f, Xs)
    v = map(f, Xs)
    X_elite = Xs[sortperm(v)[1:opt.n_elite]]
    μ_elite = mean(X_elite)
    Σ_elite = cov(X_elite)
    d = MvNormal(μ_elite, Σ_elite)
    return [rand(d) for _ in 1:opt.n], d
end

function optimize_info(opt::CrossEntropyOptimization, f, d0; iter=100)
    Xs = [rand(d0) for _ in 1:opt.n]
    d_hist = [d0]
    Xs_hist = [Xs]
    for _ ∈ 1:iter
        Xs, d = opt_iter(opt, f, Xs)
        push!(Xs_hist, Xs)
        push!(d_hist, d)
    end
    return Xs_hist, d_hist
end

opt = CrossEntropyOptimization(10, 100)
d0 = MvNormal([0.0, 0.0], diagm([3.0, 3.0]))
Xs_hist, d_hist = optimize_info(opt, camel, d0; iter=10)

function basis(i::Int,n::Int)
    x = zeros(n)
    x[i] = 1
    return x
end

function hooke_jeeves(f, x, α, ϵ, γ=0.5; iter=100)
    y, n = f(x), length(x)
    y_hist = [y]
    x_hist = [x]
    i = 0
    while α > ϵ && i < iter
        improved = false
        x_best, y_best = x, y
        for i in 1 : n
            for sgn in (-1,1)
                x′ = x + sgn*α*basis(i, n)
                y′ = f(x′)
                if y′ < y_best
                    x_best, y_best, improved = x′, y′, true
                end
            end
        end
        x, y = x_best, y_best
        if !improved
            α *= γ
        end
        push!(x_hist, x)
        push!(y_hist, y)
        i += 1
    end
    return x, x_hist, y_hist
end

x_best, x_hist, y_hist = hooke_jeeves(camel, [0.,0.], 1.0, 1e-3; iter=10)

mins = map(Xs_hist) do Xs
    minimum(camel, Xs)
end

p = plot(mins, lw=2, xlabel="Iteration", ylabel="Best iterate value", title="Six-Hump Camel Optimization", label="Cross Entropy")
plot!(p, y_hist, lw=2, label="Hooke Jeeves")
savefig(p, joinpath("figures", "hw3", "p1-CE-HJ-comparison.pdf"))

Xs = range(-2.0, 2.0, length=100)
Ys = range(-1.0, 1.0, length=100)

p1 = contourf(Xs, Ys, (x,y) -> camel(x,y), levels=50, colorbar=false)
_idx = 1
scatter!(p1, first.(Xs_hist[_idx]), last.(Xs_hist[_idx]), xlims=extrema(Xs), ylims=extrema(Ys))

p2 = contourf(Xs, Ys, (x,y) -> camel(x,y), levels=50, colorbar=false, yticks=nothing)
_idx = 2
scatter!(p2, first.(Xs_hist[_idx]), last.(Xs_hist[_idx]), xlims=extrema(Xs), ylims=extrema(Ys))

p3 = contourf(Xs, Ys, (x,y) -> camel(x,y), levels=50, colorbar=false, yticks=nothing)
_idx = 3
scatter!(p3, first.(Xs_hist[_idx]), last.(Xs_hist[_idx]), xlims=extrema(Xs), ylims=extrema(Ys))

p4 = contourf(Xs, Ys, (x,y) -> camel(x,y), levels=50, colorbar=false, yticks=nothing)
_idx = 4
scatter!(p4, first.(Xs_hist[_idx]), last.(Xs_hist[_idx]), xlims=extrema(Xs), ylims=extrema(Ys))

p5 = contourf(Xs, Ys, (x,y) -> camel(x,y), levels=50, colorbar=false, yticks=nothing)
_idx = 5
scatter!(p5, first.(Xs_hist[_idx]), last.(Xs_hist[_idx]), xlims=extrema(Xs), ylims=extrema(Ys))


p = plot(p1, p2, p3, p4, p5, layout=(1,5), size=(1500, 250))
savefig(p, joinpath("figures", "hw3", "p1-cross-entropy-contour.pdf"))

blank = scatter([], [], zcolor=[NaN], clims=(-1,6), label="", colorbar_title="cbar", background_color_subplot=:transparent, markerstrokecolor=:transparent, framestyle=:none, inset=bbox(0.1, 0, 0.6, 0.9, :center, :right), axis=false)
# l = @layout [grid(1, 5) a{0.05w}]
l = @layout [grid(1, 6)]
plot(p1, p2, p3, p4, p5, blank, layout=l, link=:all)

##
n_samples = 100
opt = CrossEntropyOptimization(10, n_samples)
d0 = MvNormal([0.0, 0.0], diagm([3.0, 3.0]))
trajs = []
for _ in 1:100
    try
        Xs_hist, d_hist = optimize_info(opt, camel, d0; iter=100)
        mins = map(Xs_hist) do Xs
            minimum(camel, Xs)
        end
        push!(trajs, mins)
    catch e
        @warn(e)
    end
end

_trajs = reduce(hcat, trajs)
μ = mean(_trajs, dims=2)
σ = std(_trajs, dims=2)

p = plot(μ, ribbon=σ, title="Cross-Entropy Optimization - $n_samples Samples", xlabel="Iteration", ylabel="Best iterate value")
savefig(p, joinpath("figures", "hw3", "p1-cross-entropy-100sample-errorbars.pdf"))

n_samples = 8
opt = CrossEntropyOptimization(4, n_samples)
d0 = MvNormal([0.0, 0.0], diagm([3.0, 3.0]))
trajs = []
for _ in 1:100
    try
        Xs_hist, d_hist = optimize_info(opt, camel, d0; iter=100)
        mins = map(Xs_hist) do Xs
            minimum(camel, Xs)
        end
        push!(trajs, mins)
    catch e
        @warn(e)
    end
end

_trajs8 = reduce(hcat, trajs)
μ8 = mean(_trajs8, dims=2)
σ8 = std(_trajs8, dims=2)

p = plot(μ8, ribbon=σ8, title="Cross-Entropy Optimization - $n_samples Samples", xlabel="Iteration", ylabel="Best iterate value")
savefig(p, joinpath("figures", "hw3", "p1-cross-entropy-100sample-errorbars.pdf"))

p = plot(μ, ribbon=σ, lw=2, title="", xlabel="Iteration", ylabel="Best iterate value", label="CE 100 samples")
plot!(p, μ8, lw=2, ribbon=σ8, label="CE 4 samples")
plot!(p, y_hist, lw=2, label="Hooke Jeeves")

savefig(p, joinpath("figures", "hw3", "p1-all-comparison.pdf"))
