# =====================================================================
#  cache.jl — per-stage binary solution cache (§10)
#  Format: Int64 DOF count, then that many Float64.  No serialization lib.
# =====================================================================

using Printf

cache_path(dir, prefix, V) = joinpath(dir, @sprintf("%s_V%.6f.bin", prefix, V))

function save_solution(dir::AbstractString, prefix::AbstractString,
                       u::Vector{Float64}, V::Float64; quiet::Bool = true)
    mkpath(dir)
    fn = cache_path(dir, prefix, V)
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    quiet || @printf("  [CACHED] %s (%d DOFs)\n", fn, length(u))
    return fn
end

function load_solution(dir::AbstractString, prefix::AbstractString, V::Float64)
    fn = cache_path(dir, prefix, V)
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

"§10.6 list all cached voltages for a stage."
function list_cache(dir::AbstractString, prefix::AbstractString)
    isdir(dir) || return Float64[]
    files = filter(f -> startswith(f, prefix * "_V") && endswith(f, ".bin"), readdir(dir))
    Vs = sort([parse(Float64, f[length(prefix)+3:end-4]) for f in files])
    isempty(Vs) || @printf("Cache: %d solutions, V in [%.3f, %.3f]\n", length(Vs), Vs[1], Vs[end])
    return Vs
end

"§10.7 stale-cache warning."
function cache_banner(stage_label, dir)
    print_kinetics(stage_label)
    @printf("Cache dir: %s\n", dir)
    if isdir(dir) && !isempty(readdir(dir))
        @printf("WARNING: Existing cache found. Verify kinetics match before reusing!\n")
    end
    flush(stdout)
end
