# cache.jl — Per-stage raw-binary solution cache (§10.3). No serialization libs.
# Format: Int64 DOF count, then Float64 solution vector. Filename encodes V to 6 dp.

using Printf

"Save solution u at voltage V to <cache_dir>/<prefix>V%.6f.bin"
function save_solution(cache_dir::String, prefix::String, u::Vector{Float64}, V::Float64)
    mkpath(cache_dir)
    fn = joinpath(cache_dir, @sprintf("%sV%.6f.bin", prefix, V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
    return fn
end

"Load solution at voltage V, or `nothing` if the file does not exist."
function load_solution(cache_dir::String, prefix::String, V::Float64)
    fn = joinpath(cache_dir, @sprintf("%sV%.6f.bin", prefix, V))
    isfile(fn) || return nothing
    open(fn, "r") do f
        n = read(f, Int64)
        read!(f, Vector{Float64}(undef, n))
    end
end

"List cached voltages (sorted) for a given cache dir + prefix."
function list_cache(cache_dir::String, prefix::String)
    isdir(cache_dir) || return Float64[]
    files = filter(f -> startswith(f, prefix*"V") && endswith(f, ".bin"), readdir(cache_dir))
    pre = length(prefix) + 1   # strip "<prefix>V" ... actually strip prefix then leading "V"
    Vs = Float64[]
    for f in files
        # filename: <prefix>V<value>.bin  → strip prefix+1 ('V') and trailing ".bin"
        s = f[length(prefix)+2 : end-4]
        push!(Vs, parse(Float64, s))
    end
    return sort(Vs)
end
