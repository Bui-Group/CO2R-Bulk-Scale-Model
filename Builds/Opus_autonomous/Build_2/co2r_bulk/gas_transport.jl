# gas_transport.jl — Stefan-Maxwell (mixture-average) + Darcy, §6
#
# The gas domain is the CONCATENATION  [GDL cells 1..N_GDL ; CL cells 1..N_CL],
# so combined cell index k runs 1..N_GDL+N_CL and faces f run 1..N_GDL+N_CL+1.
#   face 1            : flow-field inlet (Dirichlet ω_inlet, p_total, ghost cell)
#   face N_GDL+1      : GDL/CL interface (harmonic-mean D_eff and κ)
#   face N_GDL+N_CL+1 : electrolyte side (zero species flux, zero Darcy flux)

"""
Precomputed per-face geometry and effective transport coefficients for the
combined GDL+CL gas domain.
"""
struct GasGeom
    NG::Int                       # N_GDL
    NC::Int                       # N_CL
    NK::Int                       # NG + NC   (cells)
    dx::Vector{Float64}           # NK   cell widths
    dxc::Vector{Float64}          # NK+1 face-to-adjacent-centre distances
    Dbf::Array{Float64,3}         # 4×4×(NK+1) effective binary diffusivities
    kap::Vector{Float64}          # NK+1 face permeabilities
end

function make_gas_geom(cl::Mesh1D, gdl::Mesh1D)
    NG, NC = gdl.N, cl.N
    NK = NG + NC
    dx  = vcat(gdl.dx, cl.dx)
    dxc = Vector{Float64}(undef, NK + 1)
    dxc[1] = gdl.dxc[1]
    for f in 2:NG
        dxc[f] = gdl.dxc[f]
    end
    dxc[NG+1] = gdl.dxc[NG+1] + cl.dxc[1]      # GDL centre NG → CL centre 1
    for i in 1:(NC-1)
        dxc[NG+1+i] = cl.dxc[i+1]
    end
    dxc[NK+1] = cl.dxc[NC+1]

    Dg = eps_GDL^1.5 .* D_bin_raw
    Dc = eps_G^1.5   .* D_bin_raw
    Dbf = zeros(4, 4, NK + 1)
    kap = zeros(NK + 1)
    for f in 1:(NK+1)
        if f <= NG
            Dbf[:, :, f] .= Dg;  kap[f] = kappa_GDL
        elseif f == NG + 1
            for i in 1:4, j in 1:4
                (i == j) && continue
                Dbf[i, j, f] = 2 * Dg[i, j] * Dc[i, j] / (Dg[i, j] + Dc[i, j])
            end
            kap[f] = 2 * kappa_GDL * kappa_CL / (kappa_GDL + kappa_CL)
        else
            Dbf[:, :, f] .= Dc;  kap[f] = kappa_CL
        end
    end
    return GasGeom(NG, NC, NK, dx, dxc, Dbf, kap)
end

"H₂O closure: ω₄ = 1 − ω₁ − ω₂ − ω₃ (§6.1)"
@inline omega_H2O_closure(w1::Real, w2::Real, w3::Real) = max(1.0 - w1 - w2 - w3, 0.0)

"Mixture molar mass [kg/mol] from mass fractions"
@inline function Mn_from_omega(w1::Real, w2::Real, w3::Real, w4::Real)
    s = w1/M_gas[1] + w2/M_gas[2] + w3/M_gas[3] + w4/M_gas[4]
    return 1.0 / max(s, 1e-30)
end

"Mole fractions from mass fractions and the mixture molar mass"
@inline omega_to_x(wj::Real, Mn::Real, j::Int) = wj * Mn / M_gas[j]

"""
    sm_face_flux_full(wL, wR, pL, pR, geom, f, use_darcy)

Mixture-average Stefan-Maxwell + (optional) Darcy convective face flux.
`wL`, `wR` are 4-tuples of mass fractions.  Returns `(J1, J2, J3, rhou)` with
species mass fluxes [kg/m²/s] and the Darcy total mass flux [kg/m²/s].
"""
@inline function sm_face_flux_full(wL::NTuple{4,ElT}, wR::NTuple{4,ElT},
                                   pL::ElT, pR::ElT, geom::GasGeom, f::Int,
                                   use_darcy::Bool) where {ElT<:Real}
    dxc = geom.dxc[f]
    w1 = 0.5*(wL[1]+wR[1]); w2 = 0.5*(wL[2]+wR[2])
    w3 = 0.5*(wL[3]+wR[3]); w4 = 0.5*(wL[4]+wR[4])
    pf = 0.5*(pL + pR)
    MnL = Mn_from_omega(wL[1], wL[2], wL[3], wL[4])
    MnR = Mn_from_omega(wR[1], wR[2], wR[3], wR[4])
    Mnf = Mn_from_omega(w1, w2, w3, w4)
    rho = pf * Mnf / (R_gas * T)

    x1 = omega_to_x(w1, Mnf, 1); x2 = omega_to_x(w2, Mnf, 2)
    x3 = omega_to_x(w3, Mnf, 3); x4 = omega_to_x(w4, Mnf, 4)

    uG = use_darcy ? -(geom.kap[f] / mu_gas) * (pR - pL) / dxc : zero(ElT)
    rhou = rho * uG
    gradMn = (MnR - MnL) / (Mnf * dxc)

    # mixture-average diffusivity D_j^MA = (1−x_j) / Σ_{k≠j} x_k/D_jk
    J1 = _sm_one(1, x1, x2, x3, x4, wL[1], wR[1], w1, rho, geom, f, dxc, gradMn, uG)
    J2 = _sm_one(2, x2, x1, x3, x4, wL[2], wR[2], w2, rho, geom, f, dxc, gradMn, uG)
    J3 = _sm_one(3, x3, x1, x2, x4, wL[3], wR[3], w3, rho, geom, f, dxc, gradMn, uG)
    return (J1, J2, J3, rhou)
end

# helper: species j with its own mole fraction xj and the other three xa,xb,xc
@inline function _sm_one(j::Int, xj::Real, xa::Real, xb::Real, xc::Real,
                         wLj::Real, wRj::Real, wfj::Real, rho::Real,
                         geom::GasGeom, f::Int, dxc::Real, gradMn::Real, uG::Real)
    ia, ib, ic = j == 1 ? (2, 3, 4) : (j == 2 ? (1, 3, 4) : (1, 2, 4))
    denom = xa / geom.Dbf[j, ia, f] + xb / geom.Dbf[j, ib, f] + xc / geom.Dbf[j, ic, f]
    DMA = (1.0 - xj) / max(denom, 1e-12)
    return -rho * DMA * (wRj - wLj) / dxc - rho * DMA * wfj * gradMn + rho * wfj * uG
end
