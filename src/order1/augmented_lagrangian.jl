Base.@kwdef struct AugmentedLagrangeOptimizer{SOL}
    λ::Union{Float64, AbstractVector}   = 0.0
    μ::Float64                          = 0.5
    ρ::Float64                          = 1.1
    max_iter::Int                       = 100
    inner_sol::SOL                      = NewtonIteration()
end

# ḡ
augmented_inequality(g, λ, μ) = max.(g, λ ./ μ)

function augmented_lagrangian(f, g, λ, μ)
    return function (x)
        gx = g(x)
        # gax = augmented_inequality(gx, λ, μ)
        # return f(x) + dot(λ, gax) + 0.5 * μ * sum(abs2, gax)
        μ̄ = max.(gx, 0.0) .* float.(λ .> 0)
        return f(x) + dot(λ .+ 0.5 .* μ̄ .* gx, gx)
    end
end

function optimize_info(opt::AugmentedLagrangeOptimizer, f, g, x; verbose=false)
    (;λ, μ, ρ) = opt
    gx = g(x)

    if λ isa Float64
        λ = fill(λ, length(gx))
    end

    x_hist = [x]
    f_hist = [f(x)]
    g_hist = [gx]
    λ_hist = [λ]
    i = 0
    while i < opt.max_iter
        f̂ = augmented_lagrangian(f, g, λ, μ)
        _fx, x = optimize(opt.inner_sol, f̂, x)
        gx = g(x)
        λ = max.(λ .+ μ .* gx, 0.0)
        μ *= ρ
        fx = f(x)
        push!(x_hist, x)
        push!(f_hist, fx)
        push!(g_hist, gx)
        push!(λ_hist, λ)
        i += 1
    end
    return last(f_hist), last(x_hist), (;
        x = x_hist,
        f = f_hist,
        g = g_hist,
        λ = λ_hist
    )
end


