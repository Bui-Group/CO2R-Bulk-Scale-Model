# cache.jl — per-stage binary solution cache (§10). Raw binary, no serialization libs.
# Format: Int64 DOF count, then the Float64 solution vector.

using Printf

function save_solution(cache_dir::AbstractString, prefix::AbstractString,
                       u::Vector{Float64}, V::Float64)
    mkpath(cache_dir)
    fn = joinpath(cache_dir, @sprintf("%sV%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(cache_dir::AbstractString, prefix::AbstractString, V::Float64)
    fn = joinpath(cache_dir, @sprintf("%sV%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

# Load the cached solution nearest to V_target (for warm-starting).
function load_nearest(cache_dir::AbstractString, prefix::AbstractString, V_target::Float64)
    isdir(cache_dir) || return nothing, NaN
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(cache_dir))
    isempty(files) && return nothing, NaN
    Vs = [parse(Float64, f[length(prefix)+2:end-4]) for f in files]   # strip "<prefix>V" and ".bin"
    idx = argmin(abs.(Vs .- V_target))
    u = open(joinpath(cache_dir, files[idx]), "r") do f
        n = read(f, Int64); read!(f, Vector{Float64}(undef, n))
    end
    return u, Vs[idx]
end

function list_cache(cache_dir::AbstractString, prefix::AbstractString)
    isdir(cache_dir) || return Float64[]
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(cache_dir))
    sort([parse(Float64, f[length(prefix)+2:end-4]) for f in files])
end

nothing
