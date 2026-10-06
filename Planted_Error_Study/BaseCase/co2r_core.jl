# =============================================================================
# co2r_core.jl — load all core physics + solver modules in dependency order.
# Each run_stageN.jl includes this once.
# =============================================================================
include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("gas_transport.jl")
include("cache.jl")
include("solver.jl")
include("assembly.jl")
