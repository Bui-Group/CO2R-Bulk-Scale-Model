##############################################################################
# layouts.jl — global DOF vector layouts for each stage (§2, §11 Stage 5)
##############################################################################

abstract type AbstractLayout end

struct LiquidLayout <: AbstractLayout   # Stages 1-2: liquid only
    N_CL::Int
    N_TOT::Int
end
LiquidLayout(N_CL::Int) = LiquidLayout(N_CL, 5*N_CL)

struct GasLayout <: AbstractLayout      # Stages 3-4 (+ Fit): liquid + gas
    N_CL::Int
    N_GDL::Int
    N_TOT::Int
end
GasLayout(N_CL::Int, N_GDL::Int) = GasLayout(N_CL, N_GDL, 5*N_CL + 4*(N_GDL + N_CL))

struct Stage5Layout <: AbstractLayout   # Stage 5: liquid + gas + phi_s + phi_L
    N_CL::Int
    N_GDL::Int
    N_TOT::Int
end
function Stage5Layout(N_CL::Int, N_GDL::Int)
    N_TOT = 5*N_CL + 4*(N_GDL + N_CL) + N_GDL + N_CL + N_CL
    Stage5Layout(N_CL, N_GDL, N_TOT)
end

const GasCapableLayout = Union{GasLayout, Stage5Layout}

ndof(ly::AbstractLayout) = ly.N_TOT

# ── Liquid species: u[5*(ix-1)+i] for species i at CL cell ix (all layouts) ─
@inline liquid_idx(ix::Int, i::Int) = 5*(ix-1) + i

# ── Gas DOFs: N_GDL cells then N_CL cells, 4 DOFs each (w1,w2,w3,p) ─────────
@inline gas_start(ly::GasCapableLayout) = 5*ly.N_CL
@inline gas_gdl_idx(ly::GasCapableLayout, ig::Int, j::Int) = gas_start(ly) + 4*(ig-1) + j
@inline gas_cl_idx(ly::GasCapableLayout, ix::Int, j::Int)  = gas_start(ly) + 4*ly.N_GDL + 4*(ix-1) + j

# ── phi_s (GDL then CL) and phi_L (CL only) — Stage 5 only ─────────────────
@inline phi_s_gdl_start(ly::Stage5Layout) = gas_start(ly) + 4*(ly.N_GDL + ly.N_CL)
@inline phi_s_gdl_idx(ly::Stage5Layout, ig::Int) = phi_s_gdl_start(ly) + ig
@inline phi_s_cl_idx(ly::Stage5Layout, ix::Int)  = phi_s_gdl_start(ly) + ly.N_GDL + ix
@inline phi_L_idx(ly::Stage5Layout, ix::Int)     = phi_s_gdl_start(ly) + ly.N_GDL + ly.N_CL + ix

# ── Unpacking helpers — type-generic (return eltype(u) for ForwardDiff) ────
function unpack_log_conc(u::AbstractVector, N_CL::Int)
    ElT = eltype(u)
    c = zeros(ElT, 5, N_CL)
    @inbounds for ix in 1:N_CL, i in 1:5
        c[i,ix] = exp(clamp(u[liquid_idx(ix,i)], -50.0, 50.0))
    end
    return c
end

function unpack_phi_s_gdl(u::AbstractVector, ly::Stage5Layout)
    ElT = eltype(u); v = zeros(ElT, ly.N_GDL)
    @inbounds for ig in 1:ly.N_GDL; v[ig] = u[phi_s_gdl_idx(ly, ig)]; end
    return v
end
function unpack_phi_s_cl(u::AbstractVector, ly::Stage5Layout)
    ElT = eltype(u); v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_s_cl_idx(ly, ix)]; end
    return v
end
function unpack_phi_L(u::AbstractVector, ly::Stage5Layout)
    ElT = eltype(u); v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_L_idx(ly, ix)]; end
    return v
end

"Gas state at GDL cell ig: (w1, w2, w3, p) — element type matches u."
@inline function gdl_gas(u::AbstractVector, ly::GasCapableLayout, ig::Int)
    return (u[gas_gdl_idx(ly,ig,1)], u[gas_gdl_idx(ly,ig,2)], u[gas_gdl_idx(ly,ig,3)], u[gas_gdl_idx(ly,ig,4)])
end
"Gas state at CL cell ix: (w1, w2, w3, p) — element type matches u."
@inline function cl_gas(u::AbstractVector, ly::GasCapableLayout, ix::Int)
    return (u[gas_cl_idx(ly,ix,1)], u[gas_cl_idx(ly,ix,2)], u[gas_cl_idx(ly,ix,3)], u[gas_cl_idx(ly,ix,4)])
end

# ── DOF blocks for the affine-covariant weight formula (§9.5.3 step 1) ─────
dof_blocks(ly::LiquidLayout) = [1:(5*ly.N_CL)]

function dof_blocks(ly::GasLayout)
    liq  = 1:(5*ly.N_CL)
    ggdl = (5*ly.N_CL+1):(5*ly.N_CL+4*ly.N_GDL)
    gcl  = (5*ly.N_CL+4*ly.N_GDL+1):(5*ly.N_CL+4*(ly.N_GDL+ly.N_CL))
    return [liq, ggdl, gcl]
end

function dof_blocks(ly::Stage5Layout)
    liq  = 1:(5*ly.N_CL)
    ggdl = (5*ly.N_CL+1):(5*ly.N_CL+4*ly.N_GDL)
    gcl  = (5*ly.N_CL+4*ly.N_GDL+1):(5*ly.N_CL+4*(ly.N_GDL+ly.N_CL))
    ps0  = phi_s_gdl_start(ly)
    psgdl = (ps0+1):(ps0+ly.N_GDL)
    pscl  = (ps0+ly.N_GDL+1):(ps0+ly.N_GDL+ly.N_CL)
    pl    = (ps0+ly.N_GDL+ly.N_CL+1):(ps0+ly.N_GDL+2*ly.N_CL)
    return [liq, ggdl, gcl, psgdl, pscl, pl]
end

n_log_dofs(ly::AbstractLayout) = 5*ly.N_CL
