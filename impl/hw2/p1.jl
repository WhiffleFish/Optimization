using Plots
using LinearAlgebra
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

f(x1, x2) = x1^4 + 3x1^3 + 3x2^2 - 6x1*x2 - 2x2
fx1(x1, x2) = 4x1^3 + 9x1^2 - 6x2

fx1x1(x1, x2) = 12x1^2 + 18x1
fx1x2(x1, x2) = -6
fx2x2(x1, x2) = 6

p = plot(
    range(-3.0, stop=2.0, length=100), 
    x -> fx1(x, x+(1/3))
)
hline!(p, [0.0])
savefig(p, joinpath("figures", "hw2", "partial-x.pdf"))


crit_x1 = [
    -1 - 0.5√12, -(1/4), -1 + 0.5√12
]
crit_x2 = crit_x1 .+ (1/3)

hess(x1, x2) = [
    fx1x1(x1, x2) fx1x2(x1,x2)
    fx1x2(x1,x2) fx2x2(x1, x2)
]

_i = 3
isposdef(hess(crit_x1[_i], crit_x2[_i]))

x = y = range(-3, stop=2, length=100)
p = contour(x, y, f, levels=100, lw=2)
scatter!(p,
    crit_x1, crit_x2, 
    series_annotations = text.(1:length(crit_x1), :bottom),
    ms = 7, 
    markershape = :star,
    xlabel = "x1",
    ylabel = "x2"
)
savefig(p, joinpath("figures", "hw2", "p1-contour.pdf"))
