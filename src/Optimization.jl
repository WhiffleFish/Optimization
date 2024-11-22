module Optimization

using Distributions
using Zygote
using LinearAlgebra

include("base.jl")
export optimize, optimize_info

include("penalty.jl")
export PenaltyOptimizer

include(joinpath("order0", "order0.jl"))

include(joinpath("order1", "order1.jl"))

include(joinpath("order2", "order2.jl"))

end # module Optimization
