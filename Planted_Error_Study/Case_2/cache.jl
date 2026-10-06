# ============================================================================
# cache.jl — Per-stage binary solution cache (§10.3). Raw I/O, no JLD2.
#   Format: Int64 DOF count, then Float64[] solution vector.
# ============================================================================

using Printf

function save_solution(cache_dir::AbstractString, prefix::AbstractString,
                       u::Vector{Float64}, V::Float64)
    mkpath(cache_dir)
    fn = joinpath(cache_dir, @sprintf("%s_V%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(cache_dir::AbstractString, prefix::AbstractString, V::Float64)
    fn = joinpath(cache_dir, @sprintf("%s_V%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

function list_cache(cache_dir::AbstractString, prefix::AbstractString)
    isdir(cache_dir) || return Float64[]
    files = filter(f -> startswith(f, prefix * "_V") && endswith(f, ".bin"), readdir(cache_dir))
    Vs = sort([parse(Float64, f[length(prefix)+3:end-4]) for f in files])
    return Vs
end
