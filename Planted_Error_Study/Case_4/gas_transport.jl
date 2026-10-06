# gas_transport.jl — Stefan-Maxwell (mixture-average) + Darcy gas transport (§6)
#
# Gas species order [CO₂, CO, H₂, H₂O]. CO₂/CO/H₂ are DOFs (mass fraction ω);
# H₂O is closure ω₄ = max(1 − ω₁ − ω₂ − ω₃, 0). Pressure p is the 4th DOF.
# Combined gas mesh spans GDL (cells 1..N_GDL) then CL (cells N_GDL+1..N_GDL+N_CL).
#
# Gas DOF block starts after the 5·N_CL liquid DOFs:
#   u[gas_base + 4*(g-1) + j]  for gas cell g, j∈{1,2,3}=ω, j=4=p.

# ── Fuller-Schettler-Giddings binary diffusivities [m²/s] (§6.3) ──────────────
# Prefactor 1.00e-3 with M in g/mol, v in cm³/mol, p in atm → cm²/s, then ×1e-4.
# (The guide's §8.7 table stores M in kg/mol; FSG with the 1.00e-3 prefactor needs
#  g/mol — see BUILD_DIALOGUE.md. We convert M_gas[·]*1000 here.)
function fsg_binary()
    D = zeros(4, 4)
    Mg = M_gas .* 1000.0           # kg/mol → g/mol
    T175 = T^1.75
    for i in 1:4, j in 1:4
        i == j && continue
        num = 1.00e-3 * T175 * sqrt(1/Mg[i] + 1/Mg[j])
        den = 1.0 * (v_fsg[i]^(1/3) + v_fsg[j]^(1/3))^2      # p_atm = 1
        D[i, j] = num / den * 1e-4                            # cm²/s → m²/s
    end
    return D
end
const D_bin_raw = fsg_binary()     # 4×4 symmetric, m²/s

# ── composition helpers (type-generic) ───────────────────────────────────────
@inline function Mn_from_omega(ω::NTuple{4,<:Real})
    return 1.0 / (ω[1]/M_gas[1] + ω[2]/M_gas[2] + ω[3]/M_gas[3] + ω[4]/M_gas[4])
end
@inline function omega_to_x(ω::NTuple{4,<:Real})
    Mn = Mn_from_omega(ω)
    return (ω[1]/M_gas[1]*Mn, ω[2]/M_gas[2]*Mn, ω[3]/M_gas[3]*Mn, ω[4]/M_gas[4]*Mn)
end

# Mixture-average diffusivity for species j at mole fractions x, Bruggeman `brugg`.
@inline function D_MA(j::Int, x::NTuple{4,<:Real}, brugg::Real)
    s = zero(eltype(x))
    @inbounds for k in 1:4
        k == j && continue
        s += x[k] / (brugg * D_bin_raw[j, k])
    end
    return (1 - x[j]) / max(s, 1e-30)
end

# Stefan-Maxwell mixture-average face flux for CO₂,CO,H₂ [kg/m²/s] (§6.1).
# u_G = Darcy velocity at the face (0 in Stage 3; nonzero in Stage 4).
function sm_flux(ωL::NTuple{4,<:Real}, ωR::NTuple{4,<:Real},
                 pL::Real, pR::Real, brugg::Real, dxcc::Real, u_G::Real)
    xL = omega_to_x(ωL); xR = omega_to_x(ωR)
    MnL = Mn_from_omega(ωL); MnR = Mn_from_omega(ωR)
    ρL = pL * MnL / (R_gas * T); ρR = pR * MnR / (R_gas * T)
    ρbar = 0.5 * (ρL + ρR)
    Mnbar = 0.5 * (MnL + MnR)
    xbar = (0.5*(xL[1]+xR[1]), 0.5*(xL[2]+xR[2]), 0.5*(xL[3]+xR[3]), 0.5*(xL[4]+xR[4]))
    gradMn_over_Mn = (MnR - MnL) / dxcc / Mnbar
    J1 = _smj(1, ωL, ωR, xbar, ρbar, brugg, dxcc, gradMn_over_Mn, u_G)
    J2 = _smj(2, ωL, ωR, xbar, ρbar, brugg, dxcc, gradMn_over_Mn, u_G)
    J3 = _smj(3, ωL, ωR, xbar, ρbar, brugg, dxcc, gradMn_over_Mn, u_G)
    return (J1, J2, J3)
end
@inline function _smj(j, ωL, ωR, xbar, ρbar, brugg, dxcc, gradMn_over_Mn, u_G)
    DjMA = D_MA(j, xbar, brugg)
    ωbar = 0.5 * (ωL[j] + ωR[j])
    gradω = (ωR[j] - ωL[j]) / dxcc
    diff = -ρbar * DjMA * gradω - ρbar * DjMA * ωbar * gradMn_over_Mn
    conv = ρbar * ωbar * u_G                       # convective (Stage 4); u_G=0 in Stage 3
    return diff + conv
end

# ── combined gas mesh (GDL ++ CL) ─────────────────────────────────────────────
struct GasMesh
    Ng::Int
    N_GDL::Int
    N_CL::Int
    dx::Vector{Float64}          # Ng cell widths
    dx_cc::Vector{Float64}       # Ng+1 face center-distances
    brugg::Vector{Float64}       # Ng+1 face Bruggeman factors
    kappa_face::Vector{Float64}  # Ng+1 face permeabilities (Darcy, Stage 4)
    x_centers::Vector{Float64}   # Ng cell centers [m] (for plotting)
end

function build_gas_mesh(gdlm::Mesh, clm::Mesh)
    Ng = gdlm.N + clm.N
    dx = vcat(gdlm.dx, clm.dx)
    xc = vcat(gdlm.x_centers, clm.x_centers)
    dx_cc = zeros(Ng + 1)
    for f in 1:gdlm.N; dx_cc[f] = gdlm.dx_cc[f]; end
    dx_cc[gdlm.N + 1] = gdlm.dx_cc[gdlm.N + 1] + clm.dx_cc[1]     # GDL/CL interface
    for f in 2:clm.N; dx_cc[gdlm.N + f] = clm.dx_cc[f]; end
    dx_cc[Ng + 1] = clm.dx_cc[clm.N + 1]

    bg = eps_GDL^1.5; bc = eps_G^1.5; bi = 2*bg*bc/(bg + bc)
    brugg = fill(bc, Ng + 1)
    for f in 1:gdlm.N; brugg[f] = bg; end
    brugg[gdlm.N + 1] = bi

    # Darcy face permeabilities (harmonic mean at interface); Stage 4 uses these.
    ki = 2*kappa_GDL*kappa_CL/(kappa_GDL + kappa_CL)
    kappa_face = fill(kappa_CL, Ng + 1)
    for f in 1:gdlm.N; kappa_face[f] = kappa_GDL; end
    kappa_face[gdlm.N + 1] = ki
    return GasMesh(Ng, gdlm.N, clm.N, dx, dx_cc, brugg, kappa_face, xc)
end

# ── gas DOF accessors (type-generic) ──────────────────────────────────────────
@inline gas_base(N_CL::Int) = 5 * N_CL
@inline gidx(N_CL::Int, g::Int, j::Int) = gas_base(N_CL) + 4*(g - 1) + j
@inline function gas_omega(u::AbstractVector{<:Real}, N_CL::Int, g::Int)
    b = gas_base(N_CL) + 4*(g - 1)
    ω1 = u[b+1]; ω2 = u[b+2]; ω3 = u[b+3]
    ω4 = max(1 - ω1 - ω2 - ω3, zero(ω1))
    return (ω1, ω2, ω3, ω4)
end
@inline gas_p(u::AbstractVector{<:Real}, N_CL::Int, g::Int) = u[gas_base(N_CL) + 4*(g-1) + 4]

# local CO₂ partial pressure [atm] from a gas cell
@inline function pCO2_atm(u::AbstractVector{<:Real}, N_CL::Int, g::Int)
    ω = gas_omega(u, N_CL, g); x = omega_to_x(ω)
    return x[1] * gas_p(u, N_CL, g) / 101325.0
end

const omega_inlet_tuple = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])

nothing
