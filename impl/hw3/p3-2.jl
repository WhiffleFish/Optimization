using Optimization
using LinearAlgebra
using Plots
default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])


traj(u) = accumulate(+, map(uk -> 6sin(uk), u))
full_traj(u) = [0;traj(u)]
car_objective(y) = dot(y,y)
g_control(u) = [u .- 0.1; -u .- 0.1]
_y_l = [fill(-2.0, 4); fill(1.0, 6)]
_y_u = fill(2.0, 10)
g_traj(y) = [y .- _y_u;  -y .+ _y_l]
ts = 0.0:0.2:2.0
function car_objective_constraints(u)
    y = traj(u)
    return car_objective(y), vcat(g_traj(y), g_control(u))
end

function full_traj_xy(u)
    x = map(0:10) do x
        6x
    end
    y = full_traj(u)
    return x,y
end

opt = PenaltyOptimizer(NesterovMomentum(α=1e-7, max_iter=1000), max_iter=1000)

f, g, x, info = optimize_info(opt, car_objective_constraints, ones(10)*0.00)

p_violation = plot(maximum.(info.g), lw=5, title="Maximum Constraint Violation", xlabel="Iteration", ylabel="max(g)")
p_objective = plot(info.f, lw=5, title="Objective Value", ylabel="J(u) (m^2)")
p_solve = plot(p_objective, p_violation, layout=(2,1))
savefig(p_solve, joinpath("figures", "hw3", "p3-solve.pdf"))

p_traj = plot(rectangle(30, 1, 30, 0), opacity=0.5, fill=nothing, xlims=(0,60), ylims=(-0.1, 1.1))
plot!(p_traj, full_traj_xy(x)..., xlabel="x", ylabel="y", lw=5, title="Optimized Vehicle Trajectory")

p_control = plot(ts, [last(info.x); 0.0], xlabel="time (s)", ylabel="u(t)", lw=5)
p_traj_control = plot(p_traj, p_control, layout=(2,1))
savefig(p_traj_control, joinpath("figures", "hw3", "p3-traj-control.pdf"))
