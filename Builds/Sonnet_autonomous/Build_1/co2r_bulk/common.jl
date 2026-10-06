##############################################################################
# common.jl — include all shared modules in dependency order
##############################################################################
include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("gas_transport.jl")
include("solid_transport.jl")
include("layouts.jl")
include("assembly.jl")
include("newton_affine.jl")
include("cache.jl")
include("driver.jl")
include("postprocess.jl")
