# =====================================================================
#  gas_transport.jl — Stefan-Maxwell (mixture-average) + Darcy (§6)
#  Gas cells are numbered globally: g = 1..N_GDL are GDL cells,
#  g = N_GDL+1 .. N_GDL+N_CL are CL cells.  Faces run 1..Ng+1 with face 1 at
#  the flow-field inlet, face N_GDL+1 at the GDL/CL interface, face Ng+1 at
#  the electrolyte side.
# =====================================================================

"Per-face gas geometry and material properties (fixed for a given mesh)."
struct GasGeom
    N_GDL::Int
    N_CL::Int
    Ng::Int
    dx::Vector{Float64}                 # Ng cell widths
    dcc::Vector{Float64}                # Ng+1 face-normal distances
    Dbin::Vector{Matrix{Float64}}       # Ng+1 effective binary diffusivity matrices
    kappa::Vector{Float64}              # Ng+1 face permeabilities
end

function build_gas_geom(gdl::Mesh1D, cl::Mesh1D)
    Ng = gdl.N + cl.N
    dx  = vcat(gdl.dx, cl.dx)
    dcc = Vector{Float64}(undef, Ng + 1)
    Db  = Vector{Matrix{Float64}}(undef, Ng + 1)
    kap = Vector{Float64}(undef, Ng + 1)

    D_int = [ (D_bin_GDL[i,j] == 0.0 || D_bin_CL[i,j] == 0.0) ? 0.0 :
              2.0*D_bin_GDL[i,j]*D_bin_CL[i,j] / (D_bin_GDL[i,j]+D_bin_CL[i,j])
              for i in 1:4, j in 1:4 ]                       # harmonic mean at interface

    dcc[1] = gdl.dx_between[1];                Db[1] = D_bin_GDL; kap[1] = kappa_GDL
    for k in 2:gdl.N
        dcc[k] = gdl.dx_between[k];            Db[k] = D_bin_GDL; kap[k] = kappa_GDL
    end
    ki = gdl.N + 1
    dcc[ki] = gdl.dx_between[gdl.N+1] + cl.dx_between[1]
    Db[ki]  = D_int;  kap[ki] = kappa_interface                # §11 Stage 4 item 3
    for m in 2:cl.N
        dcc[ki+m-1] = cl.dx_between[m];        Db[ki+m-1] = D_bin_CL; kap[ki+m-1] = kappa_CL
    end
    dcc[Ng+1] = cl.dx_between[cl.N+1];         Db[Ng+1] = D_bin_CL;   kap[Ng+1] = kappa_CL
    return GasGeom(gdl.N, cl.N, Ng, dx, dcc, Db, kap)
end

# ── mixture helpers (all type-generic for ForwardDiff, §9.3.2) ───────
"H₂O closure: ω₄ = max(1 − ω₁ − ω₂ − ω₃, 0) (§6.1)."
@inline omega_H2O_closure(w1::ElT, w2::ElT, w3::ElT) where {ElT<:Real} =
    max(one(ElT) - w1 - w2 - w3, zero(ElT))

"Mixture molar mass M_n = 1 / Σ(ω_j/M_j) [kg/mol]."
@inline function Mn_from_omega(w::NTuple{4,ElT}) where {ElT<:Real}
    s = zero(ElT)
    @inbounds for j in 1:4; s += w[j] / M_gas[j]; end
    return one(ElT) / s
end

"Mass fractions → mole fractions."
@inline omega_to_x(w::NTuple{4,ElT}, Mn::ElT) where {ElT<:Real} =
    ntuple(j -> w[j] * Mn / M_gas[j], Val(4))

"Mixture-average diffusivity D_j^MA = (1−x_j)/Σ_{k≠j}(x_k/D_jk_eff) (§6.1)."
@inline function D_MA_species(x::NTuple{4,ElT}, Db::Matrix{Float64}, j::Int) where {ElT<:Real}
    den = zero(ElT)
    @inbounds for k in 1:4
        k == j && continue
        den += x[k] / Db[j, k]
    end
    return (one(ElT) - x[j]) / max(den, 1e-300)
end

"""
    sm_face_flux_full(wL, pL, wR, pR, Db, kappa, dcc, darcy)
        -> (J_CO2, J_CO, J_H2, rho_u)

Stefan-Maxwell mixture-average mass fluxes [kg/(m²·s)] plus the Darcy total
mass flux ρu at one face (§6.1–6.2).  `darcy = false` sets u_G = 0 (Stage 3).
"""
@inline function sm_face_flux_full(wL::NTuple{4,ElT}, pL::ElT,
                                   wR::NTuple{4,ElT}, pR::ElT,
                                   Db::Matrix{Float64}, kap::Float64,
                                   dcc::Float64, darcy::Bool) where {ElT<:Real}
    wf  = ntuple(j -> 0.5 * (wL[j] + wR[j]), Val(4))
    pf  = 0.5 * (pL + pR)
    MnL = Mn_from_omega(wL); MnR = Mn_from_omega(wR); Mnf = Mn_from_omega(wf)
    xf  = omega_to_x(wf, Mnf)
    rho = pf * Mnf / (R_gas * T)
    gMn = (MnR - MnL) / dcc
    uG  = darcy ? -(kap / mu_gas) * (pR - pL) / dcc : zero(ElT)
    Js  = ntuple(Val(4)) do j
        Dj = D_MA_species(xf, Db, j)
        gw = (wR[j] - wL[j]) / dcc
        -rho * Dj * gw - rho * Dj * wf[j] * (gMn / Mnf) + rho * wf[j] * uG
    end
    return (Js[1], Js[2], Js[3], rho * uG)
end
