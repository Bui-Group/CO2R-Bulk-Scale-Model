# co2r.jl — load all shared modules in dependency order.
using Printf, LinearAlgebra, SparseArrays

const HERE = @__DIR__

include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "gas_transport.jl"))
include(joinpath(HERE, "solver.jl"))
include(joinpath(HERE, "cache.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "assembly_stage5.jl"))

nothing
