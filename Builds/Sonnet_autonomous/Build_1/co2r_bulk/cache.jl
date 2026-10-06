##############################################################################
# cache.jl — per-stage binary solution cache (§10)
##############################################################################
using Printf

function save_solution(cache_dir::String, prefix::String, u::Vector{Float64}, V::Float64)
    mkpath(cache_dir)
    fn = joinpath(cache_dir, @sprintf("%s_V%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

function load_solution(cache_dir::String, prefix::String, V::Float64)
    fn = joinpath(cache_dir, @sprintf("%s_V%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    return open(fn, "r") do f
        n = read(f, Int64)
        v = Vector{Float64}(undef, n)
        read!(f, v)
        v
    end
end

function list_cache_voltages(cache_dir::String, prefix::String)
    isdir(cache_dir) || return Float64[]
    files = filter(f -> startswith(f, prefix*"_V") && endswith(f, ".bin"), readdir(cache_dir))
    isempty(files) && return Float64[]
    return sort([parse(Float64, f[length(prefix)+3:end-4]) for f in files])
end
