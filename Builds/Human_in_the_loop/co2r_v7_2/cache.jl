# =============================================================================
# cache.jl — per-stage binary solution cache (§10)
# Raw binary I/O only: Int64 DOF count then the Float64 solution vector.
# No JLD2, no serialization library (§10.3).
# =============================================================================
using Printf

cache_dir_for(stage::Int) = joinpath(@__DIR__, "output_stage$(stage)", "cache")
cache_prefix_for(stage::Int) = "s$(stage)_"

function save_solution(cache_dir::String, prefix::String, u::Vector{Float64}, V::Float64;
                       quiet::Bool = true)
    mkpath(cache_dir)
    fn = joinpath(cache_dir, @sprintf("%sV%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    quiet || (@printf("  [CACHED] %s (%d DOFs)\n", fn, length(u)); flush(stdout))
    return fn
end

function load_solution(cache_dir::String, prefix::String, V::Float64)
    fn = joinpath(cache_dir, @sprintf("%sV%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    return open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

"§10.6 List all cached voltages for a stage."
function list_cache(cache_dir::String, prefix::String)
    isdir(cache_dir) || return Float64[]
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(cache_dir))
    isempty(files) && return Float64[]
    Vs = sort([parse(Float64, f[length(prefix)+2:end-4]) for f in files])
    @printf("Cache: %d solutions, V in [%.3f, %.3f]\n", length(Vs), Vs[1], Vs[end])
    return Vs
end

"§10.7 Stale-cache detection banner, printed at the top of every stage driver."
function print_stage_header(stage, cache_dir::String; extra::String = "")
    println("="^78)
    @printf("=== Stage %s ===\n", string(stage))
    @printf("i0_CO2R = %.4f A/m^2, lambda_MHC = %.4f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.6e, alpha_c = %.6f, k_GL = %.3e m/s\n",
            HER_prefactor(), alpha_c_HER(), k_GL())
    @printf("Mesh: N_CL = %d, N_GDL = %d\n", N_CL, N_GDL)
    @printf("Cache dir: %s\n", cache_dir)
    if isdir(cache_dir) && !isempty(readdir(cache_dir))
        @printf("WARNING: Existing cache found. Verify kinetics match before reusing!\n")
    end
    isempty(extra) || println(extra)
    println("="^78)
    flush(stdout)
end
