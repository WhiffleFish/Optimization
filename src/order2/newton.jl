Base.@kwdef struct NewtonIteration
    max_iter::Int   = 1000
    ϵ::Float64      = 1e-3
    α::Float64      = 1.0
end

function optimize_info(opt::NewtonIteration, f, x, ∇f = zygote_grad_func(f), H = zygote_hess_func(f))
    x_hist = [x]
    f_hist = [f(x)]
    i = 0
    Δ = fill(Inf, length(x))
    while i < opt.max_iter && norm(Δ) > opt.ϵ
        hess = H(x)
        Δ = if isposdef(hess)
            hess \ ∇f(x)
        else
            ∇f(x)
        end
        x = x .- opt.α .* Δ
        push!(x_hist, x)
        push!(f_hist, f(x))
        i += 1
    end
    return last(f_hist), last(x_hist), (;
        x = x_hist,
        f = f_hist
    )
end

function optimize(opt::NewtonIteration, f, x, ∇f = zygote_grad_func(f))
    min_f, min_x, hist = optimize_info(opt, f, x, ∇f)
    return min_f, min_x
end

zygote_hess_func(f) = x -> hessian(f, x)
