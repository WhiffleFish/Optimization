using JuMP
using HiGHS
using Ipopt
using Printf


v = [1.7, 6, 4, 1]
c = [1, 2, 3, 1.5]
sigma = [0.3, 2.5, 2.0, 2.0]
Sigma = [
    0.09  0.0  0.0  0.0;
	0.0  6.25  0.0  -4.0;
	0.0  0.0   4.0   0.0;
	0.0 -4.0  0.0   4.0;
]

model = Model(HiGHS.Optimizer)
@variable(model, x[eachindex(v)] .≥ 0)
@constraint(model, dot(x,c) ≤ 1000.0)
@objective(model, Max, dot(x, v .- c))
optimize!(model)
value.(x)
dot(value.(x), c)

clipboard(join(round.(value.(x), sigdigits=3), "\\\\"))
value(dot(x, v .- c))

map(x) do x_i
	@sprintf("%06.2f", value(x_i))
end |> x -> join(x, "\\\\") |> clipboard

##
model = Model(Ipopt.Optimizer)
@variable(model, x[eachindex(v)] .≥ 0)
@constraint(model, dot(x,c) ≤ 1000.0)
@expression(model, profit , dot(x, v .- c))
@constraint(model, 0.5*profit ≥ sqrt(sum(abs2, x .* sigma)))
@objective(model, Max, profit^2)
optimize!(model)
dot(value.(x),c)
sqrt(sum(abs2, value.(x) .* sigma))
value(profit)
value.(x)

value(dot(x, v .- c))

map(x) do x_i
	@sprintf("%06.2f", value(x_i))
end |> x -> join(x, "\\\\") |> clipboard

##
model = Model(Ipopt.Optimizer)
@variable(model, x[eachindex(v)] .≥ 0)
@constraint(model, dot(x,c) ≤ 1000.0)
@expression(model, profit , dot(x, v .- c))
@constraint(model, 0.5*profit ≥ sqrt(dot(x,Sigma,x)))
@objective(model, Max, profit^2)
optimize!(model)
dot(value.(x),c)
0.5*value(profit) - sqrt(dot(value.(x),Sigma,value.(x)))
value(profit)
value.(x)

clipboard(join(round.(value.(x), sigdigits=3), "\\\\"))
value(dot(x, v .- c))

map(x) do x_i
	@sprintf("%06.2f", value(x_i))
end |> x -> join(x, "\\\\") |> clipboard
