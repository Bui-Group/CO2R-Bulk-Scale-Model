# common.jl — load every module in dependency order.

using Printf, LinearAlgebra, SparseArrays, ForwardDiff

include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("gas_transport.jl")
include("assembly.jl")
include("newton_affine.jl")
include("driver.jl")
include("export_data.jl")

"Standard model objects at the current mesh setting."
function build_model(stage::Int)
    cl  = make_cl_mesh()
    gdl = make_gdl_mesh()
    gm  = make_gas_mesh(gdl, cl)
    ly  = Layout(cl.N, gdl.N, stage)
    c_eq = equilibrium_vector()
    return (cl = cl, gdl = gdl, gm = gm, ly = ly, c_eq = c_eq)
end

"§10.7 stale-cache banner."
function print_kinetics_header(stage, cache_dir)
    @printf("=== Stage %s ===\n", stage)
    @printf("i0_CO2R = %.4f A/m², lambda_MHC = %.4f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.6e, alpha_c = %.6f, k_GL = %.3e m/s\n",
            HER_prefactor(), alpha_c_HER(), k_GL())
    @printf("Cache dir: %s\n", cache_dir)
    if isdir(cache_dir) && !isempty(readdir(cache_dir))
        @printf("WARNING: existing cache found — verify kinetics match before reusing!\n")
    end
    flush(stdout)
end

"Equilibrium liquid guess: every CL cell at the bulk equilibrium (§9.6)."
function u0_liquid(c_eq::Vector{Float64}, N_CL::Int)
    u0 = zeros(5 * N_CL)
    lnc = log.(c_eq)
    for ix in 1:N_CL
        u0[5*(ix-1)+1 : 5*ix] .= lnc
    end
    return u0
end

"Full anchor guess for a stage: liquid equilibrium + inlet gas + zero potentials."
function u0_stage(ly::Layout, c_eq::Vector{Float64})
    u0 = zeros(ly.N_TOT)
    u0[1:5*ly.N_CL] .= u0_liquid(c_eq, ly.N_CL)
    if ly.n_gas > 0
        for g in 1:ly.Ng
            u0[gas_idx(ly, g, 1)] = omega_inlet[1]
            u0[gas_idx(ly, g, 2)] = omega_inlet[2]
            u0[gas_idx(ly, g, 3)] = omega_inlet[3]
            u0[gas_idx(ly, g, 4)] = p_total
        end
    end
    # φ_s and φ_L stay at 0 (Stage 5 anchor is V_c = 0, §9.5.5)
    return u0
end
