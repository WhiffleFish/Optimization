Base.@kwdef struct HookeJeeves
    α::Float64      = 1.0
    γ::Float64      = 1e-3
    ϵ::Float64      = 0.5
    max_iter::Int   = 100
end

function basis(i::Int,n::Int)
    x = zeros(n)
    x[i] = 1
    return x
end

function optimize_info(opt::HookeJeeves, f, x)
    (; α, γ, ϵ, max_iter) = opt
    y, n = f(x), length(x)
    y_hist = [y]
    x_hist = [x]
    i = 0
    while α > ϵ && i < max_iter
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
    return y, x, (;x = x_hist, f = y_hist)
end

function optimize(opt::HookeJeeves, f, x)
    min_f, min_x, hist = optimize_info(opt, f, x)
    return min_f, min_x
end
