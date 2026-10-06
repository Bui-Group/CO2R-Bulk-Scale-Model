# gas_transport.jl — Stefan-Maxwell mixture-average diffusion + Darcy flow,
# §6. DOF indexing for the combined GDL+CL gas layout (Stages 3-5).

# ── §6.3 Fuller-Schettler-Giddings binary diffusivities ──────────────────
"FSG binary diffusivity [m^2/s]. M in g/mol, v in cm^3/mol, T in K, p in atm."
@inline function fsg_binary(Mi_g::Float64, Mj_g::Float64, vi::Float64, vj::Float64,
                             Tval::Float64, p_atm::Float64)
    Dij_cm2s = 1.00e-3 * Tval^1.75 * sqrt(1/Mi_g + 1/Mj_g) / (p_atm * (vi^(1/3) + vj^(1/3))^2)
    return Dij_cm2s * 1e-4
end

const M_gas_gmol = M_gas .* 1000.0   # kg/mol -> g/mol for the FSG formula

"Raw (uncorrected) binary diffusivity matrix at p=1 atm, T=323.15K [m^2/s]."
const D_bin_raw = begin
    Dm = zeros(4, 4)
    for i in 1:4, j in 1:4
        if i != j
            Dm[i,j] = fsg_binary(M_gas_gmol[i], M_gas_gmol[j], v_diff[i], v_diff[j], T, 1.0)
        end
    end
    Dm
end

const D_eff_GDL_gas = eps_GDL^1.5 .* D_bin_raw
const D_eff_CL_gas  = eps_G^1.5   .* D_bin_raw
const D_eff_int_gas = begin
    Dm = zeros(4, 4)
    for i in 1:4, j in 1:4
        if i != j
            Dm[i,j] = 2 * D_eff_GDL_gas[i,j] * D_eff_CL_gas[i,j] / (D_eff_GDL_gas[i,j] + D_eff_CL_gas[i,j])
        end
    end
    Dm
end

const omega_inlet_tuple = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])

# ── Mixture property helpers (type-generic, §9.3.2 checklist) ────────────
@inline Mn_from_omega(omega::NTuple{4,ElT}) where {ElT<:Real} =
    1.0 / (omega[1]/M_gas[1] + omega[2]/M_gas[2] + omega[3]/M_gas[3] + omega[4]/M_gas[4])

@inline function omega_to_x(omega::NTuple{4,ElT}, Mn::ElT) where {ElT<:Real}
    return (omega[1]*Mn/M_gas[1], omega[2]*Mn/M_gas[2], omega[3]*Mn/M_gas[3], omega[4]*Mn/M_gas[4])
end

@inline rho_gas(p::Real, Mn::Real) = p * Mn / (R_gas * T)

@inline function D_MA_species(x::NTuple{4,ElT}, Dbin_eff::Matrix{Float64}, j::Int) where {ElT<:Real}
    denom = zero(ElT)
    @inbounds for k in 1:4
        if k != j
            denom += x[k] / Dbin_eff[j,k]
        end
    end
    return (1 - x[j]) / denom
end

"Mole fraction of CO2 (species 1) from a mass-fraction tuple."
@inline function mole_fraction_CO2(omega::NTuple{4,ElT}) where {ElT<:Real}
    Mn = Mn_from_omega(omega)
    return omega[1] * Mn / M_gas[1]
end

# ── §6.1 Stefan-Maxwell face flux (mass flux, kg/m^2/s) ───────────────────
"""
    sm_face_flux(omega_L, omega_R, p_L, p_R, Dbin_eff, dx_cc, u_G)

Full mixture-average SM flux for all 4 gas species at one face, including
the mixture-molar-mass correction term and (if u_G != 0) Darcy convection.
"""
function sm_face_flux(omega_L::NTuple{4,<:Real}, omega_R::NTuple{4,<:Real},
                       p_L::Real, p_R::Real, Dbin_eff::Matrix{Float64},
                       dx_cc::Real, u_G::Real)
    Mn_L = Mn_from_omega(omega_L)
    Mn_R = Mn_from_omega(omega_R)
    omega_bar = (0.5*(omega_L[1]+omega_R[1]), 0.5*(omega_L[2]+omega_R[2]),
                 0.5*(omega_L[3]+omega_R[3]), 0.5*(omega_L[4]+omega_R[4]))
    Mn_bar = Mn_from_omega(omega_bar)
    x_bar = omega_to_x(omega_bar, Mn_bar)
    p_bar = 0.5*(p_L + p_R)
    rho_bar = rho_gas(p_bar, Mn_bar)
    dMndx = (Mn_R - Mn_L) / dx_cc
    # NTuple return: element type promotes automatically (Float64 vs Dual-safe),
    # avoiding the "same ElT for both L/R" trap when one side is a Float64
    # boundary constant and the other carries ForwardDiff.Dual numbers.
    return ntuple(4) do j
        Dj = D_MA_species(x_bar, Dbin_eff, j)
        domega = (omega_R[j] - omega_L[j]) / dx_cc
        -rho_bar*Dj*domega - rho_bar*Dj*omega_bar[j]*(dMndx/Mn_bar) + rho_bar*omega_bar[j]*u_G
    end
end

"Darcy mass flux rho*u_G at a face (for the pressure/continuity residual)."
function darcy_mass_flux(p_L::Real, p_R::Real, omega_L::NTuple{4,<:Real}, omega_R::NTuple{4,<:Real},
                          kappa_face::Real, dx_cc::Real)
    Mn_bar = Mn_from_omega((0.5*(omega_L[1]+omega_R[1]), 0.5*(omega_L[2]+omega_R[2]),
                             0.5*(omega_L[3]+omega_R[3]), 0.5*(omega_L[4]+omega_R[4])))
    p_bar = 0.5*(p_L + p_R)
    rho_bar = rho_gas(p_bar, Mn_bar)
    u_G = -(kappa_face / mu_gas) * (p_R - p_L) / dx_cc
    return rho_bar * u_G, u_G
end

# ── Combined GDL+CL gas DOF layout (Stages 3-5) ──────────────────────────
# AbstractGasLayout: both the gas-only Stage 3/4 layout (GasLayout) and the
# Stage 5 layout (Stage5Layout, defined in assembly.jl, which appends the
# phi_s/phi_L potential blocks after the identical gas block) share this
# interface, since the gas DOF positions are identical in both — only what
# comes after them differs.
abstract type AbstractGasLayout end

struct GasLayout <: AbstractGasLayout
    N_CL::Int
    N_GDL::Int
end
gas_start(ly::AbstractGasLayout) = 5*ly.N_CL
gas_gdl_idx(ly::AbstractGasLayout, ig::Int, j::Int) = gas_start(ly) + 4*(ig-1) + j
gas_cl_idx(ly::AbstractGasLayout, ix::Int, j::Int)  = gas_start(ly) + 4*ly.N_GDL + 4*(ix-1) + j
n_dofs_34(ly::AbstractGasLayout) = 5*ly.N_CL + 4*(ly.N_GDL + ly.N_CL)

function gdl_omega(u::AbstractVector{ElT}, ly::AbstractGasLayout, ig::Int) where {ElT<:Real}
    w1 = u[gas_gdl_idx(ly,ig,1)]; w2 = u[gas_gdl_idx(ly,ig,2)]; w3 = u[gas_gdl_idx(ly,ig,3)]
    w4 = max(one(ElT) - w1 - w2 - w3, zero(ElT))
    return (w1, w2, w3, w4)
end
gdl_p(u::AbstractVector{<:Real}, ly::AbstractGasLayout, ig::Int) = u[gas_gdl_idx(ly,ig,4)]

function cl_omega(u::AbstractVector{ElT}, ly::AbstractGasLayout, ix::Int) where {ElT<:Real}
    w1 = u[gas_cl_idx(ly,ix,1)]; w2 = u[gas_cl_idx(ly,ix,2)]; w3 = u[gas_cl_idx(ly,ix,3)]
    w4 = max(one(ElT) - w1 - w2 - w3, zero(ElT))
    return (w1, w2, w3, w4)
end
cl_p(u::AbstractVector{<:Real}, ly::AbstractGasLayout, ix::Int) = u[gas_cl_idx(ly,ix,4)]

function make_u0_gas!(u0::Vector{Float64}, ly::AbstractGasLayout)
    for ig in 1:ly.N_GDL
        u0[gas_gdl_idx(ly,ig,1)] = omega_inlet[1]
        u0[gas_gdl_idx(ly,ig,2)] = omega_inlet[2]
        u0[gas_gdl_idx(ly,ig,3)] = omega_inlet[3]
        u0[gas_gdl_idx(ly,ig,4)] = p_total
    end
    for ix in 1:ly.N_CL
        u0[gas_cl_idx(ly,ix,1)] = omega_inlet[1]
        u0[gas_cl_idx(ly,ix,2)] = omega_inlet[2]
        u0[gas_cl_idx(ly,ix,3)] = omega_inlet[3]
        u0[gas_cl_idx(ly,ix,4)] = p_total
    end
    return u0
end

"""
    sm_all_fluxes(u, ly, gdl_mesh, cl_mesh, u_G_faces) -> J[4, N_GDL+N_CL+1]

Assemble the combined GDL+CL Stefan-Maxwell (+ optional Darcy convection)
face-flux array in the integrated-FV convention (§6.4). `u_G_faces` is a
length-(N_GDL+N_CL+1) vector of Darcy velocities (all zero for Stage 3,
computed from the pressure DOFs for Stage 4+).
"""
function sm_all_fluxes(u::AbstractVector{ElT}, ly::AbstractGasLayout, gdl_mesh::Mesh1D, cl_mesh::Mesh1D,
                        u_G_faces::AbstractVector{<:Real}) where {ElT<:Real}
    Ntot = ly.N_GDL + ly.N_CL
    J = zeros(ElT, 4, Ntot+1)
    setcol!(col, tup) = (@inbounds for j in 1:4; J[j,col] = tup[j]; end)
    # left boundary: Dirichlet inlet composition + p_total
    om_gdl1 = gdl_omega(u, ly, 1)
    setcol!(1, sm_face_flux(omega_inlet_tuple, om_gdl1, p_total, gdl_p(u,ly,1),
                             D_eff_GDL_gas, gdl_mesh.dx_face_left, u_G_faces[1]))
    for f in 2:ly.N_GDL
        omL = gdl_omega(u,ly,f-1); omR = gdl_omega(u,ly,f)
        setcol!(f, sm_face_flux(omL, omR, gdl_p(u,ly,f-1), gdl_p(u,ly,f),
                                 D_eff_GDL_gas, gdl_mesh.dx_between[f-1], u_G_faces[f]))
    end
    # GDL/CL interface face
    omL = gdl_omega(u,ly,ly.N_GDL); omR = cl_omega(u,ly,1)
    dx_int = gdl_mesh.dx_face_right + cl_mesh.dx_face_left
    setcol!(ly.N_GDL+1, sm_face_flux(omL, omR, gdl_p(u,ly,ly.N_GDL), cl_p(u,ly,1),
                                      D_eff_int_gas, dx_int, u_G_faces[ly.N_GDL+1]))
    for f in 2:ly.N_CL
        omL = cl_omega(u,ly,f-1); omR = cl_omega(u,ly,f)
        setcol!(ly.N_GDL+f, sm_face_flux(omL, omR, cl_p(u,ly,f-1), cl_p(u,ly,f),
                                          D_eff_CL_gas, cl_mesh.dx_between[f-1], u_G_faces[ly.N_GDL+f]))
    end
    # right boundary: zero flux for species (gas doesn't penetrate electrolyte)
    # (J[:, Ntot+1] already zero-initialized)
    return J
end

# ── §7.4 combined van't Hoff + Sechenov Henry constant ────────────────────
"""
    sechenov_H_CO2(c_ix) -> H_eff [mol/(m3 atm)]

Combined T- and ion-corrected Henry constant at the local CL composition
`c_ix = (c_H, c_OH, c_HCO3, c_CO3, c_CO2)`, §7.4.
"""
@inline function sechenov_H_CO2(c_ix::AbstractVector{<:Real})
    c_Cs = cCs_cell(c_ix)
    c_OH, c_HCO3, c_CO3 = c_ix[2], c_ix[3], c_ix[4]
    exponent = -((h_CO2+h_Cs)*c_Cs + (h_CO2+h_OH)*c_OH + (h_CO2+h_HCO3)*c_HCO3 + (h_CO2+h_CO3)*c_CO3) / 1000.0
    H_vh = H_CO2_0 * exp(dH_sol * (1/T - 1/298.15))
    return H_vh * 10.0^exponent
end
