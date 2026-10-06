# model.jl — loads all model modules in dependency order.
const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "gas_transport.jl"))
include(joinpath(HERE, "assembly_gas.jl"))
include(joinpath(HERE, "assembly_stage5.jl"))
include(joinpath(HERE, "solver_sparse.jl"))
include(joinpath(HERE, "sweep.jl"))
