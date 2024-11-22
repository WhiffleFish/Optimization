using LinearAlgebra
using Zygote
using Optimization
using Plots
using LaTeXStrings
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

traj(u) = accumulate(+, map(uk -> 6sin(uk), u))
full_traj(u) = [0;traj(u)]
car_objective(y) = dot(y,y)
g_control(u) = [u .- 0.1; -u .- 0.1]
_y_l = [fill(-2.0, 4); fill(1.0, 6)]
_y_u = fill(2.0, 10)
g_traj(y) = [y .- _y_u;  -y .+ _y_l]
ts = 0.0:0.2:2.0

function full_traj_xy(u)
    x = map(0:10) do x
        6x
    end
    y = full_traj(u)
    return x,y
end

function car_objective_constraints(u)
    y = traj(u)
    return car_objective(y), vcat(g_traj(y), g_control(u))
end

f = u -> first(car_objective_constraints(u))
g = u -> last(car_objective_constraints(u))

x = u0 = zeros(10)
sol = AugmentedLagrangeOptimizer(max_iter=210, inner_sol=NewtonIteration(α=1e-5), μ=0.05, ρ=1.025)
fx, x, hist = optimize_info(sol, f, g, u0)

plot(traj(x))
plot(x)
plot(reduce(hcat, hist.λ)')

plot(last(hist.λ))
last(hist.λ)[21:30] |> plot
plot(g(last(hist.x))[21:30])

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

p_violation = plot(maximum.(hist.g), lw=5, title="Maximum Constraint Violation", xlabel="Iteration", ylabel="max(g)")
p_objective = plot(hist.f, lw=5, title="Objective Value", ylabel="J(u) (m^2)")
p_solve = plot(p_objective, p_violation, layout=(2,1))
savefig(p_solve, joinpath("figures", "hw4", "p1-solve.pdf"))

p_traj = plot(rectangle(30, 1, 30, 0), opacity=0.5, fill=nothing, xlims=(0,60), ylims=(-0.1, 1.1))
plot!(p_traj, full_traj_xy(x)..., xlabel="x", ylabel="y", lw=5, title="Optimized Vehicle Trajectory")

p_control = plot(ts, [last(hist.x); 0.0], xlabel="time (s)", ylabel="u(t)", lw=5)
p_traj_control = plot(p_traj, p_control, layout=(2,1))
savefig(p_traj_control, joinpath("figures", "hw4", "p1-traj-control.pdf"))

# lagrange multipliers
λf = last(hist.λ)
λru, λrl = λf[1:10], λf[11:20]
λcu, λcl = λf[21:30], λf[31:40]

p_lagrange = plot(
    plot([λrl λru], ylabel=L"\lambda_{r}", label=["lower bound" "upper bound"]),
    plot([λcl λcu], ylabel=L"\lambda_{c}"),
    lw = 2,
    suptitle = "Optimal Control Lagrange Multipliers"
)
savefig(p_lagrange, joinpath("figures", "hw4", "p1-lagrange-multipliers.pdf"))

# animation
for i ∈ eachindex(hist.x)
    sleep(0.1)
    display(plot(traj(hist.x[i]), title="$i"))
end
plot(last(hist.x))
x
