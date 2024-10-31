struct PenaltyOptimizer{OPT}
    inner_opt::OPT
    max_iter::Int
    γ::Float64
    ρ::Float64
    function PenaltyOptimizer(opt::OPT; max_iter=100, γ=1.1, ρ=0.1) where OPT
        return new{OPT}(opt, max_iter, γ, ρ)
    end
end

p_quadratic(gs::AbstractArray) = sum(gs) do g_i
    max(g_i, 0)^2
end

function penalty_objective(fg, ρ)
    return function (u)
        J,g = fg(u)
        return J + ρ * p_quadratic(g)
    end
end

function optimize_info(opt::PenaltyOptimizer, fg, x)
    (;inner_opt, max_iter, γ, ρ) = opt
    x_hist = Vector{Float64}[]
    penalty_hist = Float64[]
    f_hist = Float64[]
    g_hist = Vector{Float64}[]
    for i ∈ 1:max_iter
        f = penalty_objective(fg, ρ)
        f_x, x = optimize(inner_opt, f, x)
        _f, _g = fg(x)
        push!(penalty_hist, f_x)
        push!(f_hist, _f)
        push!(g_hist, _g)
        push!(x_hist, x)
        if all(≤(0), _g)
            break
        end
        ρ *= γ
    end
    return last(f_hist), last(g_hist), last(x_hist), (;
        f = f_hist,
        g = g_hist,
        p = penalty_hist,
        x = x_hist
    )
end
