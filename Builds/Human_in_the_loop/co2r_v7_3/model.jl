# ============================================================================
# model.jl — single include point for the whole model.
# ============================================================================
include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("layout.jl")
include("gas_transport.jl")
include("assembly.jl")
include("newton_affine.jl")
include("driver.jl")
include("io_export.jl")

"Print the active kinetics for stale-cache detection (§10.7)."
function print_kinetics(stage)
    @printf("=== Stage %s ===\n", string(stage))
    @printf("i0_CO2R = %.4f A/m^2, lambda_MHC = %.4f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.6e, alpha_c = %.6f, k_GL = %.3e m/s\n",
            HER_prefactor(), alpha_c_HER(), k_GL())
end

"Load fitted kinetics written by run_fit.jl (§11 Stage 5 gotcha #14)."
function load_fitted_kinetics!(path = joinpath(@__DIR__, "output_fit", "fitted_params.txt"))
    isfile(path) || error("fitted parameters not found at $path — run run_fit.jl first")
    d = Dict{String,Float64}()
    for line in eachline(path)
        (isempty(strip(line)) || startswith(line, "#")) && continue
        k, v = split(line, "=")
        d[strip(k)] = parse(Float64, strip(v))
    end
    set_i0_CO2R!(d["i0_CO2R"]); set_lambda_MHC!(d["lambda_MHC"])
    set_HER_prefactor!(d["HER_prefactor"]); set_alpha_c_HER!(d["alpha_c_HER"])
    haskey(d, "k_GL") && set_k_GL!(d["k_GL"])
    return d
end
