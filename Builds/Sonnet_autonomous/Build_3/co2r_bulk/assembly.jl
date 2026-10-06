# assembly.jl — full_residual! assembly for each stage + current extractors
# §11 Implementation Stages. Built incrementally: Stage 1/2 (liquid-only)
# first, Stage 3/4 (gas DOFs) and Stage 5 (potentials) appended later in
# this same file as the build progresses through the guide's stop gates.

# ── Shared liquid-only DOF indexing (Stages 1-2: 5*N_CL DOFs) ────────────
@inline liquid_idx(ix::Int, i::Int) = 5*(ix - 1) + i

function unpack_log_conc(u::AbstractVector{ElT}, N::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

"Cs+ recovered from electroneutrality, §2 / §3.1."
@inline cCs_cell(c_ix::AbstractVector{<:Real}) = c_ix[2] + c_ix[3] + 2*c_ix[4] - c_ix[1] - C_fix*eps_I

"Stage 1-2 GDL-side Henry's law CO2(aq) Dirichlet BC (van't Hoff only, no Sechenov), §7.1."
const c_CO2_henry_s12 = H_eff_vantHoff(T) * (x_CO2_inlet * 1.0)   # p_CO2_inlet = 0.88 atm

function make_u0_liquid_eq(c_eq_vec::Vector{Float64}, N::Int)
    u0 = zeros(5*N)
    ln_c_eq = log.(c_eq_vec)
    for ix in 1:N
        u0[5*(ix-1)+1:5*ix] .= ln_c_eq
    end
    return u0
end

"""
    residual_liquid12!(F, u, V_cathode_RHE, cl_mesh, c_eq_vec; bc=:dirichlet)

Shared Stage 1 / Stage 2 residual core. `bc = :dirichlet` (Stage 1, §11)
uses the constant electrolyte Dirichlet; `bc = :sherwood` (Stage 2, §7.2)
replaces the right-face flux with the Sherwood-Reynolds convective BC.
Simplified kinetics throughout: phi_s_local = V_applied_SHE, phi_L = 0.
"""
function residual_liquid12!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                             V_cathode_RHE::Real, cl_mesh::Mesh1D,
                             c_eq_vec::Vector{Float64}; bc::Symbol=:dirichlet) where {ElT<:Real}
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode_RHE - V_RHE_to_SHE_offset

    J = zeros(ElT, 5, N+1)
    J[5,1] = diffusion_flux(c_CO2_henry_s12, c[5,1], D_eff[5], cl_mesh.dx_face_left)
    J[1,1] = zero(ElT); J[2,1] = zero(ElT); J[3,1] = zero(ElT); J[4,1] = zero(ElT)
    for f in 2:N
        for i in 1:5
            J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], cl_mesh.dx_between[f-1])
        end
    end
    if bc == :dirichlet
        for i in 1:5
            J[i,N+1] = diffusion_flux(c[i,N], c_eq_vec[i], D_eff[i], cl_mesh.dx_face_right)
        end
    elseif bc == :sherwood
        for i in 1:5
            J[i,N+1] = -k_MT[i] * (c_eq_vec[i] - c[i,N])
        end
    else
        error("unknown bc = $bc")
    end

    for ix in 1:N
        c_ix = @view c[:,ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end
            if i == 5; src -= r_co2r; end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * cl_mesh.dx[ix]
        end
    end
    return nothing
end

residual_stage1!(F, u, V_cathode_RHE, cl_mesh, c_eq_vec) =
    residual_liquid12!(F, u, V_cathode_RHE, cl_mesh, c_eq_vec; bc=:dirichlet)

residual_stage2!(F, u, V_cathode_RHE, cl_mesh, c_eq_vec) =
    residual_liquid12!(F, u, V_cathode_RHE, cl_mesh, c_eq_vec; bc=:sherwood)

# ── Current extractors, §17.4 (used by Stages 1-4 and the Fit step) ─────
"""
    compute_i_CO(u, phi_s, cl_mesh; phi_L=0.0) -> A/m^2

Integrated CO2R partial current. For Stages 1-4 pass `phi_s =
V_applied_SHE` (a scalar, uniform); for Stage 5 pass the local phi_s_cl
vector via `compute_i_CO_stage5`.
"""
function compute_i_CO(u::Vector{Float64}, phi_s::Float64, cl_mesh::Mesh1D; phi_L::Float64=0.0)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        r = R_CO2R(c_all[5,ix], phi_s, phi_L, c_all[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i
end

function compute_i_H2(u::Vector{Float64}, phi_s::Float64, cl_mesh::Mesh1D; phi_L::Float64=0.0)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        r = R_HER(phi_s, phi_L, c_all[1,ix], a_v_CL, n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i
end

"pH profile across the CL from a liquid-layout solution vector."
function pH_profile(u::Vector{Float64}, N::Int)
    c = unpack_log_conc(u, N)
    return [-log10(max(c[1,ix]/1000.0, 1e-30)) for ix in 1:N]
end

# ── Stage 3/4: liquid + gas monolithic residual, §11 Stage 3/4, §6, §7.3-7.4 ─
"""
    residual_stage34!(F, u, V_cathode_RHE, cl_mesh, gdl_mesh, c_eq_vec, ly;
                       darcy=false)

Shared Stage 3 / Stage 4 residual core. `darcy=false` (Stage 3) pins the
pressure DOF to `p_total` everywhere and sets Darcy velocity to zero
(diffusion-only SM); `darcy=true` (Stage 4) finalizes the true Darcy
continuity equation with harmonic-mean interface permeability and activates
convective SM transport. Simplified kinetics throughout (phi_s=V_applied_SHE,
phi_L=0) — potentials are Stage 5.
"""
function residual_stage34!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                            V_cathode_RHE::Real, cl_mesh::Mesh1D, gdl_mesh::Mesh1D,
                            c_eq_vec::Vector{Float64}, ly::GasLayout;
                            darcy::Bool=false) where {ElT<:Real}
    N_CL = cl_mesh.N; N_GDL = gdl_mesh.N
    c = unpack_log_conc(u, N_CL)
    V_applied_SHE = V_cathode_RHE - V_RHE_to_SHE_offset

    Ntot = N_GDL + N_CL
    # ── Darcy velocity at every gas face (zero in Stage 3) ────────────────
    u_G_faces = zeros(ElT, Ntot+1)
    rhou_faces = zeros(ElT, Ntot+1)
    if darcy
        u_G_faces[1] = 0.0   # Dirichlet p at the GDL inlet; velocity from the interior solve
        om_gdl1 = gdl_omega(u, ly, 1)
        rm, um = darcy_mass_flux(p_total, gdl_p(u,ly,1), omega_inlet_tuple, om_gdl1, kappa_GDL, gdl_mesh.dx_face_left)
        rhou_faces[1] = rm; u_G_faces[1] = um
        for f in 2:N_GDL
            omL = gdl_omega(u,ly,f-1); omR = gdl_omega(u,ly,f)
            rm, um = darcy_mass_flux(gdl_p(u,ly,f-1), gdl_p(u,ly,f), omL, omR, kappa_GDL, gdl_mesh.dx_between[f-1])
            rhou_faces[f] = rm; u_G_faces[f] = um
        end
        kappa_int = 2*kappa_GDL*kappa_CL/(kappa_GDL+kappa_CL)
        dx_int = gdl_mesh.dx_face_right + cl_mesh.dx_face_left
        omL = gdl_omega(u,ly,N_GDL); omR = cl_omega(u,ly,1)
        rm, um = darcy_mass_flux(gdl_p(u,ly,N_GDL), cl_p(u,ly,1), omL, omR, kappa_int, dx_int)
        rhou_faces[N_GDL+1] = rm; u_G_faces[N_GDL+1] = um
        for f in 2:N_CL
            omL = cl_omega(u,ly,f-1); omR = cl_omega(u,ly,f)
            rm, um = darcy_mass_flux(cl_p(u,ly,f-1), cl_p(u,ly,f), omL, omR, kappa_CL, cl_mesh.dx_between[f-1])
            rhou_faces[N_GDL+f] = rm; u_G_faces[N_GDL+f] = um
        end
        # right boundary: no-flow (dp/dx = 0) => u_G = 0, rhou = 0
        u_G_faces[Ntot+1] = zero(ElT); rhou_faces[Ntot+1] = zero(ElT)
    end

    Jgas = sm_all_fluxes(u, ly, gdl_mesh, cl_mesh, u_G_faces)

    # ── Faradaic + phase-transfer rates per CL cell ───────────────────────
    R_CO2R_cell = zeros(ElT, N_CL); R_HER_cell = zeros(ElT, N_CL); R_PT_cell = zeros(ElT, N_CL)
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_HER_cell[ix]  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e)
        om_cl = cl_omega(u, ly, ix)
        x_CO2_local = mole_fraction_CO2(om_cl)
        p_CO2_local = x_CO2_local * cl_p(u,ly,ix) / 101325.0
        H_local = sechenov_H_CO2(c_ix)
        R_PT_cell[ix] = a_v_GL * k_GL() * (H_local * p_CO2_local - c_ix[5])
    end

    # ── Liquid species residuals (integrated FV, §6.4) ────────────────────
    Jliq = zeros(ElT, 5, N_CL+1)
    om_cl1 = cl_omega(u, ly, 1)
    p_CO2_bc = mole_fraction_CO2(om_cl1) * cl_p(u,ly,1) / 101325.0
    c_CO2_bc = sechenov_H_CO2(@view c[:,1]) * p_CO2_bc
    Jliq[5,1] = diffusion_flux(c_CO2_bc, c[5,1], D_eff[5], cl_mesh.dx_face_left)
    Jliq[1,1] = zero(ElT); Jliq[2,1] = zero(ElT); Jliq[3,1] = zero(ElT); Jliq[4,1] = zero(ElT)
    for f in 2:N_CL
        for i in 1:5
            Jliq[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], cl_mesh.dx_between[f-1])
        end
    end
    for i in 1:5
        Jliq[i,N_CL+1] = -k_MT[i] * (c_eq_vec[i] - c[i,N_CL])
    end
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix]; end
            if i == 5; src -= R_CO2R_cell[ix]; src += R_PT_cell[ix]; end
            F[liquid_idx(ix,i)] = Jliq[i,ix] - Jliq[i,ix+1] + src * cl_mesh.dx[ix]
        end
    end

    # ── Gas residuals ──────────────────────────────────────────────────────
    for ig in 1:N_GDL
        for j in 1:3
            F[gas_gdl_idx(ly,ig,j)] = Jgas[j,ig] - Jgas[j,ig+1]
        end
        if darcy
            S_mass = zero(ElT)   # no faradaic/PT sources in the GDL
            F[gas_gdl_idx(ly,ig,4)] = rhou_faces[ig] - rhou_faces[ig+1] + S_mass * gdl_mesh.dx[ig]
        else
            F[gas_gdl_idx(ly,ig,4)] = gdl_p(u,ly,ig) - p_total
        end
    end
    for ix in 1:N_CL
        f = N_GDL + ix
        S_CO2 = -R_PT_cell[ix] * M_gas[1]
        S_CO  =  R_CO2R_cell[ix] * M_gas[2]
        S_H2  =  R_HER_cell[ix]  * M_gas[3]
        Svec = (S_CO2, S_CO, S_H2)
        for j in 1:3
            F[gas_cl_idx(ly,ix,j)] = Jgas[j,f] - Jgas[j,f+1] + Svec[j] * cl_mesh.dx[ix]
        end
        if darcy
            S_mass = S_CO2 + S_CO + S_H2
            F[gas_cl_idx(ly,ix,4)] = rhou_faces[f] - rhou_faces[f+1] + S_mass * cl_mesh.dx[ix]
        else
            F[gas_cl_idx(ly,ix,4)] = cl_p(u,ly,ix) - p_total
        end
    end
    return nothing
end

residual_stage3!(F, u, V_cathode_RHE, cl_mesh, gdl_mesh, c_eq_vec, ly) =
    residual_stage34!(F, u, V_cathode_RHE, cl_mesh, gdl_mesh, c_eq_vec, ly; darcy=false)

residual_stage4!(F, u, V_cathode_RHE, cl_mesh, gdl_mesh, c_eq_vec, ly) =
    residual_stage34!(F, u, V_cathode_RHE, cl_mesh, gdl_mesh, c_eq_vec, ly; darcy=true)

function make_u0_stage34(c_eq_vec::Vector{Float64}, cl_mesh::Mesh1D, gdl_mesh::Mesh1D, ly::GasLayout)
    N = n_dofs_34(ly)
    u0 = zeros(N)
    ln_c_eq = log.(c_eq_vec)
    for ix in 1:ly.N_CL
        u0[5*(ix-1)+1:5*ix] .= ln_c_eq
    end
    make_u0_gas!(u0, ly)
    return u0
end

"CO2R / HER partial currents from a Stage-3/4-layout solution (simplified kinetics)."
function compute_i_CO_34(u::Vector{Float64}, V_applied_SHE::Float64, cl_mesh::Mesh1D)
    compute_i_CO(u, V_applied_SHE, cl_mesh)
end
function compute_i_H2_34(u::Vector{Float64}, V_applied_SHE::Float64, cl_mesh::Mesh1D)
    compute_i_H2(u, V_applied_SHE, cl_mesh)
end

# ── Stage 5: full potentials (phi_L NP migration, phi_s solid Ohm's law) ──
# §11 Stage 5, §3.1, §3.4, §9.4 items 6-8, §9.5.5, §15.3 gotcha #9a.
struct Stage5Layout <: AbstractGasLayout
    N_CL::Int
    N_GDL::Int
    N_TOT::Int
end
function Stage5Layout(N_CL::Int, N_GDL::Int)
    N_TOT = 5*N_CL + 4*(N_GDL + N_CL) + N_GDL + N_CL + N_CL
    Stage5Layout(N_CL, N_GDL, N_TOT)
end

_phi_s_gdl_start(ly::Stage5Layout) = gas_start(ly) + 4*(ly.N_GDL + ly.N_CL)
phi_s_gdl_idx(ly::Stage5Layout, ig::Int) = _phi_s_gdl_start(ly) + ig
phi_s_cl_idx(ly::Stage5Layout, ix::Int)  = _phi_s_gdl_start(ly) + ly.N_GDL + ix
phi_L_idx(ly::Stage5Layout, ix::Int)     = _phi_s_gdl_start(ly) + ly.N_GDL + ly.N_CL + ix

function unpack_phi_s_gdl(u::AbstractVector{ElT}, ly::Stage5Layout) where {ElT<:Real}
    return ElT[u[phi_s_gdl_idx(ly,ig)] for ig in 1:ly.N_GDL]
end
function unpack_phi_s_cl(u::AbstractVector{ElT}, ly::Stage5Layout) where {ElT<:Real}
    return ElT[u[phi_s_cl_idx(ly,ix)] for ix in 1:ly.N_CL]
end
function unpack_phi_L_cl(u::AbstractVector{ElT}, ly::Stage5Layout) where {ElT<:Real}
    return ElT[u[phi_L_idx(ly,ix)] for ix in 1:ly.N_CL]
end

"""
    solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, sig_gdl, sig_cl,
                         gdl_mesh, cl_mesh) -> i_s[N_GDL+N_CL+1]

§3.4 solid-phase Ohm's law face currents. Left: ghost-cell Dirichlet BC
(NOT a cell-1 residual override — §11 Stage 5 gotcha #6). Right: Neumann
zero (electrons cannot leave the solid phase except via faradaic reaction).
GDL/CL interface: harmonic-mean conductivity.
"""
function solid_face_currents(phi_s_gdl::AbstractVector{ElT}, phi_s_cl::AbstractVector{ElT},
                              V_applied_SHE::Real, sig_gdl::Float64, sig_cl::Float64,
                              gdl_mesh::Mesh1D, cl_mesh::Mesh1D) where {ElT<:Real}
    N_GDL = length(phi_s_gdl); N_CL = length(phi_s_cl)
    i_s = zeros(ElT, N_GDL + N_CL + 1)
    i_s[1] = -sig_gdl * (phi_s_gdl[1] - V_applied_SHE) / gdl_mesh.dx_face_left
    for f in 2:N_GDL
        i_s[f] = -sig_gdl * (phi_s_gdl[f] - phi_s_gdl[f-1]) / gdl_mesh.dx_between[f-1]
    end
    sig_int = 2*sig_gdl*sig_cl / (sig_gdl + sig_cl)
    dx_int = gdl_mesh.dx_face_right + cl_mesh.dx_face_left
    i_s[N_GDL+1] = -sig_int * (phi_s_cl[1] - phi_s_gdl[N_GDL]) / dx_int
    for f in 2:N_CL
        i_s[N_GDL+f] = -sig_cl * (phi_s_cl[f] - phi_s_cl[f-1]) / cl_mesh.dx_between[f-1]
    end
    i_s[N_GDL+N_CL+1] = zero(ElT)
    return i_s
end

"""
    residual_stage5!(F, u, V_cathode_RHE, cl_mesh, gdl_mesh, c_eq_vec, ly)

Full Stage 5 residual: liquid NP transport WITH migration, Stefan-Maxwell +
Darcy gas transport (identical to Stage 4), phi_L as a monolithic DOF from
the per-cell ionic charge balance (including the Cs+ migration contribution
to i_L, §3.1/§9.4 item 8), and phi_s solid-phase Ohm's law on GDL+CL.
Kinetics use LOCAL phi_s_cl[ix], phi_L_cell[ix] (not V_applied_SHE, not 0).
`V_cathode_RHE` uses the SAME convention as every other stage (V vs RHE@pH8;
converted internally) — see §11 Stage 5 gotcha #9a.
"""
function residual_stage5!(F::AbstractVector{ElT}, u::AbstractVector{ElT}, V_cathode_RHE::Real,
                           cl_mesh::Mesh1D, gdl_mesh::Mesh1D, c_eq_vec::Vector{Float64},
                           ly::Stage5Layout) where {ElT<:Real}
    N_CL = cl_mesh.N; N_GDL = gdl_mesh.N
    c = unpack_log_conc(u, N_CL)
    phi_s_gdl  = unpack_phi_s_gdl(u, ly)
    phi_s_cl   = unpack_phi_s_cl(u, ly)
    phi_L_cell = unpack_phi_L_cl(u, ly)
    V_applied_SHE = V_cathode_RHE - V_RHE_to_SHE_offset

    # ── Darcy velocities (identical structure to Stage 4) ─────────────────
    Ntot = N_GDL + N_CL
    u_G_faces = zeros(ElT, Ntot+1); rhou_faces = zeros(ElT, Ntot+1)
    om_gdl1 = gdl_omega(u, ly, 1)
    rm, um = darcy_mass_flux(p_total, gdl_p(u,ly,1), omega_inlet_tuple, om_gdl1, kappa_GDL, gdl_mesh.dx_face_left)
    rhou_faces[1] = rm; u_G_faces[1] = um
    for f in 2:N_GDL
        omL = gdl_omega(u,ly,f-1); omR = gdl_omega(u,ly,f)
        rm, um = darcy_mass_flux(gdl_p(u,ly,f-1), gdl_p(u,ly,f), omL, omR, kappa_GDL, gdl_mesh.dx_between[f-1])
        rhou_faces[f] = rm; u_G_faces[f] = um
    end
    kappa_int = 2*kappa_GDL*kappa_CL/(kappa_GDL+kappa_CL)
    dx_int_gas = gdl_mesh.dx_face_right + cl_mesh.dx_face_left
    omL = gdl_omega(u,ly,N_GDL); omR = cl_omega(u,ly,1)
    rm, um = darcy_mass_flux(gdl_p(u,ly,N_GDL), cl_p(u,ly,1), omL, omR, kappa_int, dx_int_gas)
    rhou_faces[N_GDL+1] = rm; u_G_faces[N_GDL+1] = um
    for f in 2:N_CL
        omL = cl_omega(u,ly,f-1); omR = cl_omega(u,ly,f)
        rm, um = darcy_mass_flux(cl_p(u,ly,f-1), cl_p(u,ly,f), omL, omR, kappa_CL, cl_mesh.dx_between[f-1])
        rhou_faces[N_GDL+f] = rm; u_G_faces[N_GDL+f] = um
    end
    u_G_faces[Ntot+1] = zero(ElT); rhou_faces[Ntot+1] = zero(ElT)
    Jgas = sm_all_fluxes(u, ly, gdl_mesh, cl_mesh, u_G_faces)

    # ── Faradaic + phase-transfer rates using LOCAL phi_s, phi_L ──────────
    R_CO2R_cell = zeros(ElT, N_CL); R_HER_cell = zeros(ElT, N_CL); R_PT_cell = zeros(ElT, N_CL)
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        phi_L_local = phi_L_cell[ix]
        phi_s_local = phi_s_cl[ix]
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], phi_s_local, phi_L_local, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_HER_cell[ix]  = R_HER(phi_s_local, phi_L_local, c_ix[1], a_v_CL, n_e)
        om_cl = cl_omega(u, ly, ix)
        p_CO2_local = mole_fraction_CO2(om_cl) * cl_p(u,ly,ix) / 101325.0
        H_local = sechenov_H_CO2(c_ix)
        R_PT_cell[ix] = a_v_GL * k_GL() * (H_local * p_CO2_local - c_ix[5])
    end

    # ── Step 1: liquid NP fluxes (diffusion + migration), §3.1 / §9.1 ─────
    Jliq = zeros(ElT, 5, N_CL+1)
    om_cl1 = cl_omega(u, ly, 1)
    p_CO2_bc = mole_fraction_CO2(om_cl1) * cl_p(u,ly,1) / 101325.0
    c_CO2_bc = sechenov_H_CO2(@view c[:,1]) * p_CO2_bc
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    Jliq[5,1] = zero(ElT)   # was: diffusion_flux(c_CO2_bc, c[5,1], D_eff[5], cl_mesh.dx_face_left)
    Jliq[1,1] = zero(ElT); Jliq[2,1] = zero(ElT); Jliq[3,1] = zero(ElT); Jliq[4,1] = zero(ElT)
    for f in 2:N_CL
        for i in 1:5
            Jliq[i,f] = np_flux(c[i,f-1], c[i,f], phi_L_cell[f-1], phi_L_cell[f], D_eff[i], z_i[i], cl_mesh.dx_between[f-1])
        end
    end
    # Right face: full NP flux to the c_eq reservoir at the phi_L=0 gauge
    # point, L_eff = D_eff/k_MT (§11 Stage 5 gotcha #12 — REQUIRED, not the
    # Sherwood-only formula, or charge conservation is violated at the
    # electrolyte boundary).
    for i in 1:5
        L_eff = D_eff[i] / k_MT[i]
        Jliq[i,N_CL+1] = np_flux(c[i,N_CL], c_eq_vec[i], phi_L_cell[N_CL], zero(ElT), D_eff[i], z_i[i], L_eff)
    end

    # ── Cs+ migration flux (recovered from electroneutrality, NOT a DOF, but
    # its NP migration current MUST enter i_L — §3.1, §9.4 item 8). No flux
    # at either CL boundary face.
    J_Cs = zeros(ElT, N_CL+1)
    J_Cs[1] = zero(ElT)
    for f in 2:N_CL
        cCs_L = cCs_cell(@view c[:,f-1]); cCs_R = cCs_cell(@view c[:,f])
        J_Cs[f] = np_flux(cCs_L, cCs_R, phi_L_cell[f-1], phi_L_cell[f], D_eff_Cs, z_Cs, cl_mesh.dx_between[f-1])
    end
    J_Cs[N_CL+1] = zero(ElT)

    # ── Step 3: liquid species residuals (integrated FV) ──────────────────
    for ix in 1:N_CL
        c_ix = @view c[:,ix]
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix]; end
            if i == 5; src -= R_CO2R_cell[ix]; src += R_PT_cell[ix]; end
            F[liquid_idx(ix,i)] = Jliq[i,ix] - Jliq[i,ix+1] + src * cl_mesh.dx[ix]
        end
    end

    # ── Step 4: gas residuals (species + Darcy continuity, same as Stage 4) ─
    for ig in 1:N_GDL
        for j in 1:3
            F[gas_gdl_idx(ly,ig,j)] = Jgas[j,ig] - Jgas[j,ig+1]
        end
        F[gas_gdl_idx(ly,ig,4)] = rhou_faces[ig] - rhou_faces[ig+1]
    end
    for ix in 1:N_CL
        f = N_GDL + ix
        S_CO2 = -R_PT_cell[ix] * M_gas[1]
        S_CO  =  R_CO2R_cell[ix] * M_gas[2]
        S_H2  =  R_HER_cell[ix]  * M_gas[3]
        Svec = (S_CO2, S_CO, S_H2)
        for j in 1:3
            F[gas_cl_idx(ly,ix,j)] = Jgas[j,f] - Jgas[j,f+1] + Svec[j] * cl_mesh.dx[ix]
        end
        S_mass = S_CO2 + S_CO + S_H2
        F[gas_cl_idx(ly,ix,4)] = rhou_faces[f] - rhou_faces[f+1] + S_mass * cl_mesh.dx[ix]
    end

    # ── Step 5: phi_L residual — local ionic charge balance (§3.1) ────────
    i_L_face = zeros(ElT, N_CL+1)
    for f in 1:N_CL+1
        s = zero(ElT)
        for i in 1:5
            s += z_i[i] * Jliq[i,f]
        end
        s += z_Cs * J_Cs[f]
        i_L_face[f] = F_const * s
    end
    for ix in 1:N_CL
        i_far_vol = n_e * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_L_idx(ly,ix)] = (i_L_face[ix+1] - i_L_face[ix]) + i_far_vol * cl_mesh.dx[ix]
    end
    # Dirichlet gauge override at the electrolyte-side cell (§3.1: the
    # per-cell phi_L residual has exactly one null direction — a uniform
    # shift — removed by fixing phi_L[N_CL] = 0).
    F[phi_L_idx(ly, N_CL)] = phi_L_cell[N_CL]

    # ── Step 6: phi_s residual on GDL + CL (§3.4) ──────────────────────────
    i_s = solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, sigma_s_eff_GDL, sigma_s_eff_CL, gdl_mesh, cl_mesh)
    for ig in 1:N_GDL
        F[phi_s_gdl_idx(ly,ig)] = i_s[ig+1] - i_s[ig]
    end
    for ix in 1:N_CL
        i_far_vol = n_e * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_s_cl_idx(ly,ix)] = (i_s[N_GDL+1+ix] - i_s[N_GDL+ix]) - i_far_vol * cl_mesh.dx[ix]
    end
    return nothing
end

"V_c=0 anchor guess (§9.5.5): liquid=bulk equilibrium, gas=inlet, phi_s=phi_L=0."
function make_u0_stage5_anchor(c_eq_vec::Vector{Float64}, cl_mesh::Mesh1D, gdl_mesh::Mesh1D, ly::Stage5Layout)
    u0 = zeros(ly.N_TOT)
    ln_c_eq = log.(c_eq_vec)
    for ix in 1:ly.N_CL
        u0[5*(ix-1)+1:5*ix] .= ln_c_eq
    end
    make_u0_gas!(u0, ly)
    for ig in 1:ly.N_GDL
        u0[phi_s_gdl_idx(ly,ig)] = 0.0
    end
    for ix in 1:ly.N_CL
        u0[phi_s_cl_idx(ly,ix)] = 0.0
        u0[phi_L_idx(ly,ix)] = 0.0
    end
    return u0
end

"phi_s DOF shift fallback for the tangent predictor (§9.5.4): add dV to every phi_s DOF."
function phi_s_shift_stage5!(u::Vector{Float64}, ly::Stage5Layout, dV::Float64)
    for ig in 1:ly.N_GDL
        u[phi_s_gdl_idx(ly,ig)] += dV
    end
    for ix in 1:ly.N_CL
        u[phi_s_cl_idx(ly,ix)] += dV
    end
    return nothing
end

function compute_i_CO_stage5(u::Vector{Float64}, ly::Stage5Layout, cl_mesh::Mesh1D)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        phi_s_local = u[phi_s_cl_idx(ly,ix)]; phi_L_local = u[phi_L_idx(ly,ix)]
        r = R_CO2R(c_all[5,ix], phi_s_local, phi_L_local, c_all[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
function compute_i_H2_stage5(u::Vector{Float64}, ly::Stage5Layout, cl_mesh::Mesh1D)
    c_all = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        phi_s_local = u[phi_s_cl_idx(ly,ix)]; phi_L_local = u[phi_L_idx(ly,ix)]
        r = R_HER(phi_s_local, phi_L_local, c_all[1,ix], a_v_CL, n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
