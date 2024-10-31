Base.@kwdef struct CrossEntropy
    n_elite::Int = 10
    n::Int = 100
    max_iter::Int = 100
    ϵ::Float64 = 1e-3
end

function step(opt::CrossEntropy, f, Xs)
    v = map(f, Xs)
    X_elite = Xs[sortperm(v)[1:opt.n_elite]]
    μ_elite = mean(X_elite)
    Σ_elite = cov(X_elite)
    d = MvNormal(μ_elite, Σ_elite)
    return [rand(d) for _ in 1:opt.n], d
end

function optimize_info(opt::CrossEntropy, f, d)
    Xs = [rand(d) for _ in 1:opt.n]
    min_f, _min_x_idx = findmin(f, Xs)
    min_x = Xs[_min_x_idx]
    Xs_hist = [Xs]
    f_hist = [min_f]
    best_x_hist = [min_x]
    for _ ∈ 1:opt.max_iter
        Xs, d = step(opt, f, Xs)
        push!(Xs_hist, Xs)
        _min_f, _min_x_idx = findmin(f, Xs)
        _min_x = Xs[_min_x_idx]
        stop_trial = abs(_min_f - min_f) < opt.ϵ
        if _min_f < min_f
            min_f = _min_f
            min_x = _min_x
        end
        push!(f_hist, min_f)
        push!(best_x_hist, min_x)
        stop_trial && break
    end
    return min_f, min_x, (;
        f = f_hist,
        Xs = Xs_hist,
        x = best_x_hist
    )
end

function optimize(opt::CrossEntropy, f, d)
    min_f, min_x, hist = optimize_info(opt, f, d)
    return min_f, min_x
end
