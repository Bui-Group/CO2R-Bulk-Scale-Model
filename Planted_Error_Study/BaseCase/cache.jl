# =============================================================================
# cache.jl — per-stage binary solution cache (BaseCase.md §10)
# Raw binary I/O (NO serialization library): [Int64 DOF count][Float64 u...].
# Filename encodes voltage to 6 decimals, e.g. s1_V-0.300000.bin.
# =============================================================================

using Printf

cache_dir(stage::Int) = joinpath(@__DIR__, "output_stage$(stage)", "cache")

function save_solution(stage::Int, u::Vector{Float64}, V::Float64)
    dir = cache_dir(stage)
    mkpath(dir)
    fn = joinpath(dir, @sprintf("s%d_V%.6f.bin", stage, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(stage::Int, V::Float64)
    fn = joinpath(cache_dir(stage), @sprintf("s%d_V%.6f.bin", stage, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

# Load the cached solution for the grid voltage nearest to V_target (warm-start helper, §10.5).
function load_nearest(stage::Int, V_target::Float64)
    dir = cache_dir(stage)
    isdir(dir) || return nothing
    prefix = "s$(stage)_V"
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(dir))
    isempty(files) && return nothing
    Vs = [parse(Float64, f[length(prefix)+1:end-4]) for f in files]
    idx = argmin(abs.(Vs .- V_target))
    return load_solution(stage, Vs[idx]), Vs[idx]
end

function list_cache(stage::Int)
    dir = cache_dir(stage)
    isdir(dir) || return Float64[]
    prefix = "s$(stage)_V"
    files = filter(f -> startswith(f, prefix) && endswith(f, ".bin"), readdir(dir))
    sort([parse(Float64, f[length(prefix)+1:end-4]) for f in files])
end
