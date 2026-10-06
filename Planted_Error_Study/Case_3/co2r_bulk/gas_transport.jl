# gas_transport.jl — Stefan-Maxwell mixture-average diffusion + Darcy (§6).
# Gas domain = GDL cells (x∈[-L_GDL,0]) then CL cells (x∈[0,L_CL]), 4 DOFs/cell
# (ω_CO2, ω_CO, ω_H2, p). H2O is closure: ω4 = 1-ω1-ω2-ω3. Type-generic for ForwardDiff.

# Combined gas mesh spanning GDL then CL.
struct GasMesh
    n::Int
    N_GDL::Int
    N_CL::Int
    dx::Vector{Float64}          # cell widths, length n
    dx_between::Vector{Float64}  # center-to-center, length n-1
    dx_face_left::Float64        # inlet face (x=-L_GDL) → GDL cell1 center
    dx_face_right::Float64       # CL cellN center → outlet face (x=L_CL)
    brugg::Vector{Float64}       # per-cell Bruggeman factor ε^1.5
end
function GasMesh(gdl::FVMesh, cl::FVMesh)
    n = gdl.N + cl.N
    dx  = vcat(gdl.dx, cl.dx)
    xc  = vcat(gdl.xc, cl.xc)
    dxb = [xc[k+1]-xc[k] for k in 1:n-1]
    brugg = vcat(fill(eps_GDL^1.5, gdl.N), fill(eps_G^1.5, cl.N))
    GasMesh(n, gdl.N, cl.N, dx, dxb, gdl.dx_face_left, cl.dx_face_right, brugg)
end

# §6.3 Fuller-Schettler-Giddings binary diffusivities [m²/s] at p_total=1 atm.
# M in g/mol; prefactor 1.00e-3 (Perry's 9th ed.); ×1e-4 cm²/s → m²/s.
function fsg_Dbin()
    Mg = M_gas .* 1000.0
    D = zeros(4,4)
    for i in 1:4, j in 1:4
        i == j && continue
        D[i,j] = 1.00e-3 * T^1.75 * sqrt(1/Mg[i] + 1/Mg[j]) /
                 (1.0 * (v_diff[i]^(1/3) + v_diff[j]^(1/3))^2) * 1e-4
    end
    return D
end
const D_bin = fsg_Dbin()

# Mixture molar mass [kg/mol] and mole fractions from mass fractions (4-tuples).
Mn_from_omega(ω) = 1.0 / (ω[1]/M_gas[1] + ω[2]/M_gas[2] + ω[3]/M_gas[3] + ω[4]/M_gas[4])
function omega_to_x(ω)
    s = ω[1]/M_gas[1] + ω[2]/M_gas[2] + ω[3]/M_gas[3] + ω[4]/M_gas[4]
    return (ω[1]/M_gas[1]/s, ω[2]/M_gas[2]/s, ω[3]/M_gas[3]/s, ω[4]/M_gas[4]/s)
end

const omega_inlet_tuple = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])

"""
    sm_face_flux(ωL, pL, ωR, pR, dx_cc, brugg_face, kappa_face, advect) -> (J1,J2,J3, ρu)

Stefan-Maxwell mixture-average species mass fluxes [kg/m²/s] for CO₂,CO,H₂ (§6.1),
plus the Darcy mass flux ρu at the face. `advect=false` ⇒ u_G=0 (Stage 3).
ωL, ωR are 4-tuples (incl. H₂O closure).
"""
function sm_face_flux(ωL, pL::Real, ωR, pR::Real, dx_cc::Real,
                      brugg_face::Real, kappa_face::Real, advect::Bool)
    MnL = Mn_from_omega(ωL); MnR = Mn_from_omega(ωR)
    xL = omega_to_x(ωL);     xR = omega_to_x(ωR)
    ωf = (0.5*(ωL[1]+ωR[1]), 0.5*(ωL[2]+ωR[2]), 0.5*(ωL[3]+ωR[3]), 0.5*(ωL[4]+ωR[4]))
    xf = (0.5*(xL[1]+xR[1]), 0.5*(xL[2]+xR[2]), 0.5*(xL[3]+xR[3]), 0.5*(xL[4]+xR[4]))
    Mnf = 0.5*(MnL+MnR); pf = 0.5*(pL+pR)
    ρf = pf * Mnf / (R_gas * T)
    u_G = advect ? -(kappa_face/mu_gas) * (pR-pL)/dx_cc : zero(pf)
    grad_lnMn = (MnR - MnL)/dx_cc / Mnf
    J1 = _sm_species(1, xf, ωf, ωL, ωR, ρf, brugg_face, dx_cc, grad_lnMn, u_G)
    J2 = _sm_species(2, xf, ωf, ωL, ωR, ρf, brugg_face, dx_cc, grad_lnMn, u_G)
    J3 = _sm_species(3, xf, ωf, ωL, ωR, ρf, brugg_face, dx_cc, grad_lnMn, u_G)
    ρu = ρf * u_G
    return J1, J2, J3, ρu
end

@inline function _sm_species(j, xf, ωf, ωL, ωR, ρf, brugg_face, dx_cc, grad_lnMn, u_G)
    denom = zero(xf[1])
    @inbounds for k in 1:4
        k == j && continue
        denom += xf[k] / (brugg_face * D_bin[j,k])
    end
    DjMA = (1 - xf[j]) / denom
    gradω = (ωR[j] - ωL[j]) / dx_cc
    return -ρf*DjMA*gradω - ρf*DjMA*ωf[j]*grad_lnMn + ρf*ωf[j]*u_G
end

# Harmonic-mean face Bruggeman / permeability between adjacent cells.
harm(a, b) = 2*a*b/(a+b)

# Per-cell permeability (κ_GDL in GDL, κ_CL in CL).
kappa_cell(g, N_GDL) = g <= N_GDL ? kappa_GDL : kappa_CL
