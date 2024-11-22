using JuMP, HiGHS

B = [
    (1,1,5),
    (2,1,6),
    (4,1,8),
    (5,1,4),
    (6,1,7),

    (1,2,3),
    (3,2,9),
    (7,2,6),

    (3,3,8),

    (2,4,1),
    (5,4,8),
    (8,4,4),

    (1,5,7),
    (2,5,9),
    (4,5,6),
    (6,5,2),
    (8,5,1),
    (9,5,8),

    (2,6,5),
    (5,6,3),
    (8,6,9),

    (7,7,2),

    (3,8,6),
    (7,8,8),
    (9,8,7),

    (4,9,3),
    (5,9,1),
    (6,9,6),
    (8,9,5),
    (9,9,9)
]

sum(rand(1,2,3), dims=(3,))

model = Model(HiGHS.Optimizer)
@variable(model, X[1:9, 1:9, 1:9], Bin)
@constraint(model, sum(X, dims=1) .== 1)
@constraint(model, sum(X, dims=2) .== 1)
@constraint(model, sum(X, dims=3) .== 1)
for U ∈ (1,4,7), V ∈ (1,4,7)
    @constraint(model, sum(X[U:U+2, V:V+2, :], dims=(1,2)) .== 1)
end
for (i,j,k) ∈ B
    @constraint(model, X[i,j,k] == 1)
end
optimize!(model)

m = zeros(Int, 9, 9)
V = value.(X)
for i ∈ 1:9, j ∈ 1:9
    m[i,j] = findfirst(isone, V[i,j,:])
end

m

mapreduce(join, *, eachrow(m))
