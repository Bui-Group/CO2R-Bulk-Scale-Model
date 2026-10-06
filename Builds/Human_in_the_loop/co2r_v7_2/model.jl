# =============================================================================
# model.jl — single include point for the whole model.
# Order matters: params -> mesh -> chemistry -> kinetics -> transport ->
# gas_transport -> assembly -> newton_affine -> cache -> driver
# =============================================================================
include(joinpath(@__DIR__, "params.jl"))
include(joinpath(@__DIR__, "mesh.jl"))
include(joinpath(@__DIR__, "chemistry.jl"))
include(joinpath(@__DIR__, "kinetics.jl"))
include(joinpath(@__DIR__, "transport.jl"))
include(joinpath(@__DIR__, "gas_transport.jl"))
include(joinpath(@__DIR__, "assembly.jl"))
include(joinpath(@__DIR__, "newton_affine.jl"))
include(joinpath(@__DIR__, "cache.jl"))
include(joinpath(@__DIR__, "driver.jl"))
