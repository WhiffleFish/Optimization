Base.@kwdef struct NesterovMomentum
    α::Float64          = 1e-2
    γ::Float64          = 0.9
    v::Vector{Float64}  = Float64[]
    max_iter::Int       = 100
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

function optimize_info(opt::NesterovMomentum, f, x, ∇f = zygote_grad_func(f))
    x_hist = [x]
    f_hist = [f(x)]
    initialize!(opt, x)
    for i ∈ 1:opt.max_iter
        x = step!(opt, f, ∇f, x)
        push!(x_hist, x)
        push!(f_hist, f(x))
    end
    return last(f_hist), last(x_hist), (;
        x = x_hist,
        f = f_hist
    )
end

function optimize(opt::NesterovMomentum, f, x, ∇f = zygote_grad_func(f))
    min_f, min_x, hist = optimize_info(opt, f, x, ∇f)
    return min_f, min_x
end

zygote_grad_func(f) = x -> first(gradient(f, x))
