# ============================================================================
# gas_transport.jl — Stefan-Maxwell (mixture-average) + Darcy gas transport (§6),
# Sechenov-corrected Henry constant (§7.4), combined GDL+CL gas mesh, and
# gas DOF indexing for Stages 3–5.
#   gas species index: 1 CO₂, 2 CO, 3 H₂, 4 H₂O (closure)
# ============================================================================

# ── §6.3 Fuller-Schettler-Giddings binary diffusivities (m²/s at p=1 atm, T) ─
function fsg_Dbin(i::Int, j::Int)
    Mi = M_gas[i] * 1000.0      # g/mol  (FSG uses g/mol)
    Mj = M_gas[j] * 1000.0
    D_cm2s = 1.00e-3 * T^1.75 * sqrt(1/Mi + 1/Mj) /
             (1.0 * (v_diff[i]^(1/3) + v_diff[j]^(1/3))^2)   # p_atm = 1
    return D_cm2s * 1e-4        # → m²/s
end
const D_bin_raw = [i == j ? 0.0 : fsg_Dbin(i, j) for i in 1:4, j in 1:4]

@inline harmonic_mean(a, b) = 2a * b / (a + b)

# ── Combined gas mesh spanning [−L_GDL, L_CL] (GDL cells then CL cells) ──────
function make_gas_mesh(gdl::Mesh, cl::Mesh)
    N = gdl.N + cl.N
    x_faces   = vcat(gdl.x_faces[1:end-1], cl.x_faces)       # drop duplicate x=0
    x_centers = vcat(gdl.x_centers, cl.x_centers)
    dx        = vcat(gdl.dx, cl.dx)
    dx_between = vcat(gdl.dx_between,
                      cl.x_centers[1] - gdl.x_centers[end],   # interface gap
                      cl.dx_between)
    return Mesh(N, x_faces, x_centers, dx, dx_between, gdl.dx_face_left, cl.dx_face_right)
end

# ── Gas mixture helpers (type-generic) ──────────────────────────────────────
@inline function omega_full(w1::Real, w2::Real, w3::Real)
    w4 = max(1 - w1 - w2 - w3, zero(w1))
    return (w1, w2, w3, w4)
end
@inline Mn_from_omega(w) = 1 / (w[1]/M_gas[1] + w[2]/M_gas[2] + w[3]/M_gas[3] + w[4]/M_gas[4])
@inline function omega_to_x(w)
    s = w[1]/M_gas[1] + w[2]/M_gas[2] + w[3]/M_gas[3] + w[4]/M_gas[4]
    return ((w[1]/M_gas[1])/s, (w[2]/M_gas[2])/s, (w[3]/M_gas[3])/s, (w[4]/M_gas[4])/s)
end
@inline rho_gas(p::Real, Mn::Real) = p * Mn / (R_gas * T)

# Inlet mass-fraction tuple (closure included)
gas_inlet_tuple() = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])

# Stefan-Maxwell mixture-average face flux [kg/m²/s] for the 4 species.
# `bf` = Bruggeman factor at the face (harmonic mean of ε^1.5 of the two cells).
# `dx_cc` = center-to-center distance. u_G = 0 here (Stage 3); convection added
# separately in Stage 4 via the +ρ ω u_G term.
@inline function sm_face_flux(wL, wR, p_face::Real, bf::Real, dx_cc::Real)
    wf = (0.5*(wL[1]+wR[1]), 0.5*(wL[2]+wR[2]), 0.5*(wL[3]+wR[3]), 0.5*(wL[4]+wR[4]))
    xf = omega_to_x(wf)
    MnL = Mn_from_omega(wL); MnR = Mn_from_omega(wR); Mnf = Mn_from_omega(wf)
    rhof = rho_gas(p_face, Mnf)
    gradMn_over_Mn = (MnR - MnL) / (dx_cc * Mnf)
    # Mixture-average diffusivity per species (×bf), then flux
    J = ntuple(4) do j
        denom = zero(xf[1])
        @inbounds for k in 1:4
            k == j && continue
            denom += xf[k] / D_bin_raw[j, k]
        end
        DMA = bf * (1 - xf[j]) / denom
        -rhof * DMA * (wR[j] - wL[j]) / dx_cc - rhof * DMA * wf[j] * gradMn_over_Mn
    end
    return J
end

# Stefan-Maxwell flux WITH Darcy convection (Stage 4+). Returns (J::NTuple{4},
# rho_u) where rho_u = ρ·u_G is the total convective mass flux at the face.
# Species flux J_j = diffusive SM + ρ ω_j u_G  (§6.1 full form).
@inline function sm_face_flux_conv(wL, wR, p_face::Real, bf::Real, dx_cc::Real, u_G::Real)
    wf = (0.5*(wL[1]+wR[1]), 0.5*(wL[2]+wR[2]), 0.5*(wL[3]+wR[3]), 0.5*(wL[4]+wR[4]))
    xf = omega_to_x(wf)
    MnL = Mn_from_omega(wL); MnR = Mn_from_omega(wR); Mnf = Mn_from_omega(wf)
    rhof = rho_gas(p_face, Mnf)
    gradMn_over_Mn = (MnR - MnL) / (dx_cc * Mnf)
    J = ntuple(4) do j
        denom = zero(xf[1])
        @inbounds for k in 1:4
            k == j && continue
            denom += xf[k] / D_bin_raw[j, k]
        end
        DMA = bf * (1 - xf[j]) / denom
        diff = -rhof * DMA * (wR[j] - wL[j]) / dx_cc - rhof * DMA * wf[j] * gradMn_over_Mn
        conv = rhof * wf[j] * u_G
        diff + conv
    end
    return J, rhof * u_G
end

# ── §7.4 Sechenov-corrected combined Henry constant [mol/(m³·atm)] ──────────
# c = (c_H, c_OH, c_HCO3, c_CO3, c_CO2). Cs⁺ from electroneutrality.
@inline function sechenov_H_CO2(c)
    cCs = c_Cs_from_EN(c)
    vant = H_CO2_0 * exp(dH_sol * (1/T - 1/298.15))
    sech_exp = -((h_CO2 + h_Cs)*cCs + (h_CO2 + h_OH)*c[2] +
                 (h_CO2 + h_HCO3)*c[3] + (h_CO2 + h_CO3)*c[4]) / 1000.0
    return vant * 10.0^(sech_exp)
end

# ── Gas DOF layout for Stages 3–5 ───────────────────────────────────────────
struct GasLayout
    N_CL::Int
    N_GDL::Int
end
@inline ndof_stage3(ly::GasLayout) = 5*ly.N_CL + 4*(ly.N_GDL + ly.N_CL)

# Base offset of the 4 gas DOFs for combined gas cell c (1..N_GDL = GDL, then CL).
@inline function _gas_base(ly::GasLayout, c::Int)
    return c <= ly.N_GDL ? 5*ly.N_CL + 4*(c-1) :
                           5*ly.N_CL + 4*ly.N_GDL + 4*(c - ly.N_GDL - 1)
end
@inline gas_resid_idx(ly::GasLayout, c::Int, j::Int) = _gas_base(ly, c) + j

@inline function gas_cell_omega(u::AbstractVector, ly::GasLayout, c::Int)
    b = _gas_base(ly, c)
    return omega_full(u[b+1], u[b+2], u[b+3])
end
@inline gas_cell_p(u::AbstractVector, ly::GasLayout, c::Int) = u[_gas_base(ly, c) + 4]

@inline gas_cell_eps(ly::GasLayout, c::Int) = c <= ly.N_GDL ? eps_GDL : eps_G

# Darcy permeability per combined gas cell (κ_GDL in GDL, κ_CL in CL).
@inline gas_cell_kappa(ly::GasLayout, c::Int) = c <= ly.N_GDL ? kappa_GDL : kappa_CL

# Local CO₂ partial pressure [atm] at combined gas cell c.
@inline function pCO2_atm(u::AbstractVector, ly::GasLayout, c::Int)
    w = gas_cell_omega(u, ly, c)
    x = omega_to_x(w)
    return x[1] * gas_cell_p(u, ly, c) / 101325.0
end

# ── §9.3.3 fixed structural sparsity pattern for Stages 3–4 (and the Fit step) ──
# Block layout liquid | GDL gas | CL gas. Guaranteed superset of the FV Jacobian:
#   liquid–liquid block-tridiagonal (5×5), liquid[ix] ↔ CL-gas[ix] local coupling
#   (R_PT, faradaic gas sources, CO₂ Henry left BC), gas–gas block-tridiagonal (4×4)
#   over the combined GDL+CL gas mesh. The pattern never changes, so the persistent
#   LU (§9.5.2) reuses one symbolic analysis for the whole sweep.
# (`liquid_idx` lives in assembly.jl, which every driver includes before use.)
function build_gas_pattern(ly::GasLayout)
    rows = Int[]; cols = Int[]
    add!(r, cc) = (push!(rows, r); push!(cols, cc))
    for ix in 1:ly.N_CL, jx in max(1, ix-1):min(ly.N_CL, ix+1), a in 1:5, b in 1:5
        add!(liquid_idx(ix, a), liquid_idx(jx, b))
    end
    for ix in 1:ly.N_CL, a in 1:5, b in 1:4
        add!(liquid_idx(ix, a), gas_resid_idx(ly, ly.N_GDL + ix, b))
        add!(gas_resid_idx(ly, ly.N_GDL + ix, b), liquid_idx(ix, a))
    end
    NC = ly.N_GDL + ly.N_CL
    for gc in 1:NC, gd in max(1, gc-1):min(NC, gc+1), a in 1:4, b in 1:4
        add!(gas_resid_idx(ly, gc, a), gas_resid_idx(ly, gd, b))
    end
    return rows, cols
end
