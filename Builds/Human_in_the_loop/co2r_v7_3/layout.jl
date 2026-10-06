# ============================================================================
# layout.jl — DOF layout and index functions (§2, §11 Stage 5)
#
# u = [ ln(c)_1..5N_CL ; gas_1..4(N_GDL+N_CL) ;
#       φ_s_GDL_1..N_GDL ; φ_s_CL_1..N_CL ; φ_L_CL_1..N_CL ]
# ============================================================================

struct Layout
    stage::Int
    N_CL::Int
    N_GDL::Int
    N_TOT::Int
end

function Layout(stage::Int, N_CL::Int, N_GDL::Int)
    n = 5 * N_CL
    if stage >= 3
        n += 4 * (N_GDL + N_CL)
    end
    if stage >= 5
        n += N_GDL + N_CL + N_CL
    end
    return Layout(stage, N_CL, N_GDL, n)
end

@inline liquid_idx(ix::Int, i::Int) = 5*(ix - 1) + i
@inline _gas_start(ly::Layout) = 5 * ly.N_CL
# gas cells are indexed globally: g = 1..N_GDL are GDL, g = N_GDL+1..N are CL
@inline gas_idx(ly::Layout, g::Int, j::Int) = _gas_start(ly) + 4*(g - 1) + j
@inline gas_gdl_idx(ly::Layout, ig::Int, j::Int) = gas_idx(ly, ig, j)
@inline gas_cl_idx(ly::Layout, ix::Int, j::Int)  = gas_idx(ly, ly.N_GDL + ix, j)

@inline _phi_s_start(ly::Layout) = _gas_start(ly) + 4*(ly.N_GDL + ly.N_CL)
@inline phi_s_gdl_idx(ly::Layout, ig::Int) = _phi_s_start(ly) + ig
@inline phi_s_cl_idx(ly::Layout, ix::Int)  = _phi_s_start(ly) + ly.N_GDL + ix
@inline phi_L_idx(ly::Layout, ix::Int)     = _phi_s_start(ly) + ly.N_GDL + ly.N_CL + ix

"Natural DOF blocks used for the affine-covariant weight vector (§9.5.3)."
function dof_blocks(ly::Layout)
    blks = UnitRange{Int}[1:(5*ly.N_CL)]
    if ly.stage >= 3
        g0 = _gas_start(ly)
        push!(blks, (g0 + 1):(g0 + 4*ly.N_GDL))
        push!(blks, (g0 + 4*ly.N_GDL + 1):(g0 + 4*(ly.N_GDL + ly.N_CL)))
    end
    if ly.stage >= 5
        p0 = _phi_s_start(ly)
        push!(blks, (p0 + 1):(p0 + ly.N_GDL))
        push!(blks, (p0 + ly.N_GDL + 1):(p0 + ly.N_GDL + ly.N_CL))
        push!(blks, (p0 + ly.N_GDL + ly.N_CL + 1):(p0 + ly.N_GDL + 2*ly.N_CL))
    end
    return blks
end

# ── unpacking helpers (all return the element type of `u`, §11 Stage 5 #1) ──
function unpack_log_conc(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    c = zeros(ElT, 5, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

@inline function gas_omega(u::AbstractVector{ElT}, ly::Layout, g::Int) where {ElT<:Real}
    w1 = u[gas_idx(ly, g, 1)]
    w2 = u[gas_idx(ly, g, 2)]
    w3 = u[gas_idx(ly, g, 3)]
    w4 = max(one(ElT) - w1 - w2 - w3, zero(ElT))     # §6.1 closure
    return (w1, w2, w3, w4)
end
@inline gas_p(u::AbstractVector{<:Real}, ly::Layout, g::Int) = u[gas_idx(ly, g, 4)]

function unpack_phi_s_gdl(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_GDL)
    @inbounds for ig in 1:ly.N_GDL; v[ig] = u[phi_s_gdl_idx(ly, ig)]; end
    return v
end
function unpack_phi_s_cl(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_s_cl_idx(ly, ix)]; end
    return v
end
function unpack_phi_L_cl(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_L_idx(ly, ix)]; end
    return v
end
