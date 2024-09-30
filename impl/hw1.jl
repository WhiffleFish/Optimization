using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")
f(x) = x^3 + x*cos(3x) + sin(8x^2)/2
df(x) = 3x^2 + cos(3x) - 3x*sin(3x) + 8x*cos(8x^2)

X = -1:0.001:1
Y = f.(X)
DF = df.(X)
L = ceil(maximum(DF))

# a) is function unimodal on [-1,1]
# b) is function Lipschitz on [-1,1]? What is L?
# c) find and plot crit points
# d) use shubert_piyavskii to find lower bound for function after 50 iters



plot(X,Y)
p = plot(X,DF, label="f'(x)", title="Lipschitz Justification")
hline!(p, [-L,L], label="L")
savefig(p, joinpath("figures", "hw1", "Lipschitz-Justification.pdf"))


iscrit(a,b,c) = a < b > c || a > b < c

_crits = map(eachindex(Y)[2:end-1]) do i
    iscrit(Y[i-1], Y[i], Y[i+1])
end
crits = [false;_crits;false]

p = plot(X,Y, label="f(x)", title="Critical Points")
scatter!(p, X[crits], Y[crits], label="critical points", series_annotations = text.(1:length(X[crits]), :bottom))
savefig(p, joinpath("figures", "hw1", "Critical-Points.pdf"))

struct Pt{T}
    x::T
    y::T
end

function _get_sp_intersection(A, B, l)
    t = ((A.y - B.y) - l*(A.x - B.x)) / 2l
    return Pt(A.x + t, A.y - t*l)
end

function shubert_piyavskii(f, a, b, l, ϵ; δ=0.01, max_iter=100)
    m = (a+b)/2
    A, M, B = Pt(a, f(a)), Pt(m, f(m)), Pt(b, f(b))
    pts = [A, _get_sp_intersection(A, M, l),
    M, _get_sp_intersection(M, B, l), B]
    Δ = Inf
    iter = 0
    while Δ > ϵ && iter < max_iter
        iter += 1
        i = argmin([P.y for P in pts])
        P = Pt(pts[i].x, f(pts[i].x))
        Δ = P.y - pts[i].y
        P_prev = _get_sp_intersection(pts[i-1], P, l)
        P_next = _get_sp_intersection(P, pts[i+1], l)
        deleteat!(pts, i)
        insert!(pts, i, P_next)
        insert!(pts, i, P)
        insert!(pts, i, P_prev)
    end
    intervals = []
    P_min = pts[2*(argmin([P.y for P in pts[1:2:end]])) - 1]
    y_min = P_min.y
    for i in 2:2:length(pts)
        if pts[i].y < y_min
        dy = y_min - pts[i].y
        x_lo = max(a, pts[i].x - dy/l)
        x_hi = min(b, pts[i].x + dy/l)
            if !isempty(intervals) && intervals[end][2] + δ ≥ x_lo
                intervals[end] = (intervals[end][1], x_hi)
            else
                push!(intervals, (x_lo, x_hi))
            end
        end
    end
    return P_min, intervals, (;
        pts,
        iter
    )
end

P_min, intervals, info = shubert_piyavskii(f, -1.0, 1.0, L, 1e-3, max_iter=50)

p = plot(X, Y, label="f(x)", title = "Shubert Piyavskii")
plot!(getfield.(info.pts, :x), getfield.(info.pts, :y), label="Lower Bound")
scatter!([P_min.x], [P_min.y], marker=:star, ms=10, label="Approximate Minimizer")
savefig(p, joinpath("figures", "hw1", "Shubert-Piyavskii.pdf"))


plot(X,Y)

last(X[crits]) - P_min.x

φ = 0.5 * (1 + √5)
φ^2 / 3
