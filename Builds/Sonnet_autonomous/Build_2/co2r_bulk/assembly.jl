# assembly.jl — layouts, DOF unpacking, full_residual! for every stage, current extractors.
# §2 (DOF layout), §11 (per-stage residuals).

# ── Layouts ────────────────────────────────────────────────────────────────

"Stages 1-2 (has_gas=false, N_TOT=5 N_CL) and Stages 3-4 (has_gas=true)."
struct Layout14
    N_CL::Int
    N_GDL::Int
    N_TOT::Int
    has_gas::Bool
end
function Layout14(N_CL::Int, N_GDL::Int, has_gas::Bool)
    N_TOT = has_gas ? 5*N_CL + 4*(N_GDL + N_CL) : 5*N_CL
    return Layout14(N_CL, N_GDL, N_TOT, has_gas)
end

"Stage 5: full electrochemistry (adds phi_s on GDL+CL and phi_L on CL). §11 Stage 5, verbatim."
struct Stage5Layout
    N_CL::Int; N_GDL::Int; N_TOT::Int
end
function Stage5Layout(N_CL::Int, N_GDL::Int)
    N_TOT = 5*N_CL + 4*(N_GDL + N_CL) + N_GDL + N_CL + N_CL
    return Stage5Layout(N_CL, N_GDL, N_TOT)
end

liquid_idx(ix::Int, i::Int) = 5*(ix - 1) + i

# Shared by Layout14 (Stages 3-4) and Stage5Layout -- identical formula, generic over any layout
# with N_CL/N_GDL fields. Aliased to the *_idx5 names the guide gives as "required" for Stage 5.
gas_gdl_idx(ly, ig::Int, j::Int) = 5*ly.N_CL + 4*(ig - 1) + j
gas_cl_idx(ly, ix::Int, j::Int)  = 5*ly.N_CL + 4*ly.N_GDL + 4*(ix - 1) + j
const gas_gdl_idx5 = gas_gdl_idx
const gas_cl_idx5  = gas_cl_idx

phi_s_gdl_idx5(ly::Stage5Layout, ig::Int) = 5*ly.N_CL + 4*(ly.N_GDL + ly.N_CL) + ig
phi_s_cl_idx5(ly::Stage5Layout, ix::Int)  = 5*ly.N_CL + 4*(ly.N_GDL + ly.N_CL) + ly.N_GDL + ix
phi_L_idx5(ly::Stage5Layout, ix::Int)     = 5*ly.N_CL + 4*(ly.N_GDL + ly.N_CL) + ly.N_GDL + ly.N_CL + ix

function dof_block_ranges(ly::Layout14)
    liquid = 1:(5*ly.N_CL)
    ly.has_gas || return [liquid]
    gas0 = 5*ly.N_CL
    gdl_r = (gas0+1):(gas0 + 4*ly.N_GDL)
    cl_r  = (gas0 + 4*ly.N_GDL + 1):(gas0 + 4*(ly.N_GDL + ly.N_CL))
    return [liquid, gdl_r, cl_r]
end
function dof_block_ranges(ly::Stage5Layout)
    liquid = 1:(5*ly.N_CL)
    gas0 = 5*ly.N_CL
    gdl_r = (gas0+1):(gas0 + 4*ly.N_GDL)
    cl_r  = (gas0 + 4*ly.N_GDL + 1):(gas0 + 4*(ly.N_GDL + ly.N_CL))
    ps0 = gas0 + 4*(ly.N_GDL + ly.N_CL)
    ps_gdl_r = (ps0+1):(ps0 + ly.N_GDL)
    ps_cl_r  = (ps0 + ly.N_GDL + 1):(ps0 + ly.N_GDL + ly.N_CL)
    pL0 = ps0 + ly.N_GDL + ly.N_CL
    pL_r = (pL0+1):(pL0 + ly.N_CL)
    return [liquid, gdl_r, cl_r, ps_gdl_r, ps_cl_r, pL_r]
end

# ── Unpacking helpers (return the eltype of `u` so ForwardDiff duals propagate) ────

function unpack_log_conc(u::AbstractVector{ElT}, N_CL::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N_CL)
    @inbounds for ix in 1:N_CL, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end
unpack_log_conc5(u, ly) = unpack_log_conc(u, ly.N_CL)

function cl_omega(u::AbstractVector{ElT}, ly, ix::Int) where {ElT<:Real}
    w1 = u[gas_cl_idx(ly, ix, 1)]; w2 = u[gas_cl_idx(ly, ix, 2)]; w3 = u[gas_cl_idx(ly, ix, 3)]
    return (w1, w2, w3, omega_H2O_closure(w1, w2, w3))
end
cl_p(u::AbstractVector{<:Real}, ly, ix::Int) = u[gas_cl_idx(ly, ix, 4)]

function gdl_omega(u::AbstractVector{ElT}, ly, ig::Int) where {ElT<:Real}
    w1 = u[gas_gdl_idx(ly, ig, 1)]; w2 = u[gas_gdl_idx(ly, ig, 2)]; w3 = u[gas_gdl_idx(ly, ig, 3)]
    return (w1, w2, w3, omega_H2O_closure(w1, w2, w3))
end
gdl_p(u::AbstractVector{<:Real}, ly, ig::Int) = u[gas_gdl_idx(ly, ig, 4)]

const cl_omega5 = cl_omega; const cl_p5 = cl_p
const gdl_omega5 = gdl_omega; const gdl_p5 = gdl_p

function unpack_phi_s_gdl5(u::AbstractVector{ElT}, ly::Stage5Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_GDL)
    @inbounds for ig in 1:ly.N_GDL; v[ig] = u[phi_s_gdl_idx5(ly, ig)]; end
    return v
end
function unpack_phi_s_cl5(u::AbstractVector{ElT}, ly::Stage5Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_s_cl_idx5(ly, ix)]; end
    return v
end
function unpack_phi_L_cl5(u::AbstractVector{ElT}, ly::Stage5Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_L_idx5(ly, ix)]; end
    return v
end

# ── Shared gas-phase residual assembly (§6, §11 Stages 3-5) ───────────────
#
# Builds the full GDL+CL face-flux arrays and assembles the gas-species + pressure/continuity
# residuals. `convective=false` reproduces Stage 3 (u_G pinned to 0, pressure DOF placeholder
# p==p_total); `convective=true` is Stage 4+ (real Darcy momentum balance).
function gas_residual!(F::AbstractVector{ElT}, ly, omega_gdl, p_gdl, omega_cl, p_cl,
                        R_PT_cell, R_CO2R_cell, R_HER_cell, gdl_mesh::FVMesh, cl_mesh::FVMesh;
                        convective::Bool) where {ElT<:Real}
    N_GDL = gdl_mesh.N; N_CL = cl_mesh.N
    NF = N_GDL + N_CL + 1
    omega_inlet_t = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4])

    Jgas = zeros(ElT, 4, NF)     # right boundary (col NF) stays zero: sealed CL/electrolyte face
    rho_u = zeros(ElT, NF)

    Jf, ru = sm_face_flux_full(omega_inlet_t, p_total, omega_gdl[1], p_gdl[1],
                                D_eff_GDL_mat, kappa_GDL, gdl_mesh.dx_between[1], convective)
    for j in 1:4; Jgas[j, 1] = Jf[j]; end
    rho_u[1] = ru

    for k in 2:N_GDL
        Jf, ru = sm_face_flux_full(omega_gdl[k-1], p_gdl[k-1], omega_gdl[k], p_gdl[k],
                                    D_eff_GDL_mat, kappa_GDL, gdl_mesh.dx_between[k], convective)
        for j in 1:4; Jgas[j, k] = Jf[j]; end
        rho_u[k] = ru
    end

    dx_int = gdl_cl_interface_dx(gdl_mesh, cl_mesh)
    Jf, ru = sm_face_flux_full(omega_gdl[N_GDL], p_gdl[N_GDL], omega_cl[1], p_cl[1],
                                D_eff_int_mat, kappa_interface, dx_int, convective)
    for j in 1:4; Jgas[j, N_GDL+1] = Jf[j]; end
    rho_u[N_GDL+1] = ru

    for ix in 2:N_CL
        k = N_GDL + ix
        Jf, ru = sm_face_flux_full(omega_cl[ix-1], p_cl[ix-1], omega_cl[ix], p_cl[ix],
                                    D_eff_CL_mat, kappa_CL, cl_mesh.dx_between[ix], convective)
        for j in 1:4; Jgas[j, k] = Jf[j]; end
        rho_u[k] = ru
    end
    # k = NF (right boundary): zero species flux, zero Darcy mass flux -- left at initialization.

    for ig in 1:N_GDL
        for j in 1:3
            F[gas_gdl_idx(ly, ig, j)] = Jgas[j, ig] - Jgas[j, ig+1]
        end
        F[gas_gdl_idx(ly, ig, 4)] = convective ? (rho_u[ig] - rho_u[ig+1]) : (p_gdl[ig] - p_total)
    end

    for ix in 1:N_CL
        k = N_GDL + ix
        S1 = -R_PT_cell[ix] * M_gas[1]
        S2 =  R_CO2R_cell[ix] * M_gas[2]
        S3 =  R_HER_cell[ix]  * M_gas[3]
        F[gas_cl_idx(ly, ix, 1)] = Jgas[1, k] - Jgas[1, k+1] + S1 * cl_mesh.dx[ix]
        F[gas_cl_idx(ly, ix, 2)] = Jgas[2, k] - Jgas[2, k+1] + S2 * cl_mesh.dx[ix]
        F[gas_cl_idx(ly, ix, 3)] = Jgas[3, k] - Jgas[3, k+1] + S3 * cl_mesh.dx[ix]
        if convective
            F[gas_cl_idx(ly, ix, 4)] = rho_u[k] - rho_u[k+1] + (S1+S2+S3) * cl_mesh.dx[ix]
        else
            F[gas_cl_idx(ly, ix, 4)] = p_cl[ix] - p_total
        end
    end
    return F
end

"Solid-phase (phi_s) face currents on the full GDL+CL domain, §3.4. Ghost-cell Dirichlet at the
current collector (face 1); harmonic-mean conductivity at the GDL/CL interface; Neumann zero at
the electrolyte-side boundary (last face)."
function solid_face_currents(phi_s_gdl::AbstractVector{ElT}, phi_s_cl::AbstractVector{ElT},
                              V_applied_SHE::Real, sigma_gdl::Real, sigma_cl::Real,
                              gdl_mesh::FVMesh, cl_mesh::FVMesh) where {ElT<:Real}
    N_GDL = gdl_mesh.N; N_CL = cl_mesh.N
    i_s = zeros(ElT, N_GDL + N_CL + 1)
    i_s[1] = -sigma_gdl * (phi_s_gdl[1] - V_applied_SHE) / gdl_mesh.dx_between[1]
    for ig in 2:N_GDL
        i_s[ig] = -sigma_gdl * (phi_s_gdl[ig] - phi_s_gdl[ig-1]) / gdl_mesh.dx_between[ig]
    end
    sigma_int = 2*sigma_gdl*sigma_cl/(sigma_gdl+sigma_cl)
    dx_int = gdl_cl_interface_dx(gdl_mesh, cl_mesh)
    i_s[N_GDL+1] = -sigma_int * (phi_s_cl[1] - phi_s_gdl[N_GDL]) / dx_int
    for ix in 2:N_CL
        i_s[N_GDL+ix] = -sigma_cl * (phi_s_cl[ix] - phi_s_cl[ix-1]) / cl_mesh.dx_between[ix]
    end
    i_s[N_GDL+N_CL+1] = zero(ElT)     # Neumann zero flux at the electrolyte side
    return i_s
end

# ── Stage 1 & 2: liquid NP diffusion only, simplified kinetics (phi_L=0, phi_s=V_applied) ──

"""
    residual_stage1or2!(F, u, V_cathode, cl_mesh, c_eq_vec; sherwood)

Shared implementation of Stages 1-2 (§11): identical liquid-diffusion residual, differing only in
the right (electrolyte-side) boundary flux -- Dirichlet-diffusion (Stage 1, `sherwood=false`) vs.
convective Sherwood-Reynolds Neumann (Stage 2, `sherwood=true`).
"""
function residual_stage1or2!(F::AbstractVector{ElT}, u::AbstractVector{ElT}, V_cathode::Real,
                              cl_mesh::FVMesh, c_eq_vec::Vector{Float64};
                              sherwood::Bool) where {ElT<:Real}
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c_CO2_henry = H_eff_vanthoff() * (x_inlet[1] * p_total / 101325.0)

    J = zeros(ElT, 5, N+1)
    J[5, 1] = diffusion_flux(c_CO2_henry, c[5, 1], D_eff[5], cl_mesh.dx_between[1])
    for i in 1:4; J[i, 1] = zero(ElT); end
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl_mesh.dx_between[f])
        end
    end
    for i in 1:5
        J[i, N+1] = sherwood ? (-k_MT[i] * (c_eq_vec[i] - c[i, N])) :
                                diffusion_flux(c[i, N], c_eq_vec[i], D_eff[i], cl_mesh.dx_between[N+1])
    end

    for ix in 1:N
        c_ix = view(c, :, ix)
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e_val)
        r_her  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e_val)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*r_co2r + 2*r_her)
            i == 5 && (src -= r_co2r)
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end
    return F
end

residual_stage1!(F, u, V, cl_mesh, c_eq_vec) =
    residual_stage1or2!(F, u, V, cl_mesh, c_eq_vec; sherwood=false)
residual_stage2!(F, u, V, cl_mesh, c_eq_vec) =
    residual_stage1or2!(F, u, V, cl_mesh, c_eq_vec; sherwood=true)

# ── Stages 3 & 4: + Stefan-Maxwell gas transport (+ Darcy finalized in Stage 4) ──

"""
    residual_stage34!(F, u, V_cathode, cl_mesh, gdl_mesh, c_eq_vec, ly; convective)

Shared implementation of Stages 3-4 (§11): liquid NP diffusion (still phi_L=0, phi_s=V_applied)
+ gas-phase Stefan-Maxwell transport + volumetric CO2 phase transfer. `convective=false` is
Stage 3 (u_G pinned to 0, pressure DOF placeholder); `true` is Stage 4 (Darcy finalized).
"""
function residual_stage34!(F::AbstractVector{ElT}, u::AbstractVector{ElT}, V_cathode::Real,
                            cl_mesh::FVMesh, gdl_mesh::FVMesh, c_eq_vec::Vector{Float64},
                            ly::Layout14; convective::Bool) where {ElT<:Real}
    N_CL = cl_mesh.N; N_GDL = gdl_mesh.N
    c = unpack_log_conc(u, N_CL)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    omega_gdl = [gdl_omega(u, ly, ig) for ig in 1:N_GDL]
    p_gdl     = [gdl_p(u, ly, ig)     for ig in 1:N_GDL]
    omega_cl  = [cl_omega(u, ly, ix)  for ix in 1:N_CL]
    p_cl      = [cl_p(u, ly, ix)      for ix in 1:N_CL]

    p_CO2_1 = mole_fraction_CO2(omega_cl[1]) * p_cl[1] / 101325.0
    c_CO2_ghost = sechenov_H_CO2(view(c, :, 1)) * p_CO2_1

    J = zeros(ElT, 5, N_CL+1)
    J[5, 1] = diffusion_flux(c_CO2_ghost, c[5, 1], D_eff[5], cl_mesh.dx_between[1])
    for i in 1:4; J[i, 1] = zero(ElT); end
    for f in 2:N_CL
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl_mesh.dx_between[f])
        end
    end
    for i in 1:5
        J[i, N_CL+1] = -k_MT[i] * (c_eq_vec[i] - c[i, N_CL])
    end

    R_CO2R_cell = zeros(ElT, N_CL); R_HER_cell = zeros(ElT, N_CL); R_PT_cell = zeros(ElT, N_CL)
    for ix in 1:N_CL
        c_ix = view(c, :, ix)
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], V_applied_SHE, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e_val)
        R_HER_cell[ix]  = R_HER(V_applied_SHE, 0.0, c_ix[1], a_v_CL, n_e_val)
        p_CO2_local = mole_fraction_CO2(omega_cl[ix]) * p_cl[ix] / 101325.0
        R_PT_cell[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(c_ix) * p_CO2_local - c_ix[5])
    end

    for ix in 1:N_CL
        buf = buffer_sources(view(c, :, ix))
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix])
            if i == 5
                src -= R_CO2R_cell[ix]
                src += R_PT_cell[ix]
            end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end

    gas_residual!(F, ly, omega_gdl, p_gdl, omega_cl, p_cl, R_PT_cell, R_CO2R_cell, R_HER_cell,
                  gdl_mesh, cl_mesh; convective=convective)
    return F
end

residual_stage3!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly) =
    residual_stage34!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly; convective=false)
residual_stage4!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly) =
    residual_stage34!(F, u, V, cl_mesh, gdl_mesh, c_eq_vec, ly; convective=true)

# ── Stage 5: + phi_L (NP migration) + phi_s (solid Ohm's law), fitted kinetics ──

"""
    residual_stage5!(F, u, V_cathode, cl_mesh, gdl_mesh, c_eq_vec, ly)

§11 Stage 5. Includes the Cs+ migration contribution to i_L_face (BUILD_DIALOGUE.md item 4) and
reuses that same (Cs+-inclusive) i_L_face at the right boundary as i_ionic_R for the current-
conservation diagnostic (item 4 also covers this).
"""
function residual_stage5!(F::AbstractVector{ElT}, u::AbstractVector{ElT}, V_cathode::Real,
                           cl_mesh::FVMesh, gdl_mesh::FVMesh, c_eq_vec::Vector{Float64},
                           ly::Stage5Layout) where {ElT<:Real}
    N_CL = cl_mesh.N; N_GDL = gdl_mesh.N
    c = unpack_log_conc(u, N_CL)
    phi_s_gdl  = unpack_phi_s_gdl5(u, ly)
    phi_s_cl   = unpack_phi_s_cl5(u, ly)
    phi_L_cell = unpack_phi_L_cl5(u, ly)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    omega_gdl = [gdl_omega(u, ly, ig) for ig in 1:N_GDL]
    p_gdl     = [gdl_p(u, ly, ig)     for ig in 1:N_GDL]
    omega_cl  = [cl_omega(u, ly, ix)  for ix in 1:N_CL]
    p_cl      = [cl_p(u, ly, ix)      for ix in 1:N_CL]

    # Step 1: liquid NP fluxes (diffusion + migration)
    p_CO2_1 = mole_fraction_CO2(omega_cl[1]) * p_cl[1] / 101325.0
    c_CO2_ghost = sechenov_H_CO2(view(c, :, 1)) * p_CO2_1

    J = zeros(ElT, 5, N_CL+1)
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5, 1] = zero(ElT)   # was: diffusion_flux(c_CO2_ghost, c[5, 1], D_eff[5], cl_mesh.dx_between[1])
    for i in 1:4; J[i, 1] = zero(ElT); end
    for f in 2:N_CL
        for i in 1:5
            J[i, f] = np_flux(c[i, f-1], c[i, f], phi_L_cell[f-1], phi_L_cell[f],
                               D_eff[i], z_liq[i], cl_mesh.dx_between[f])
        end
    end
    for i in 1:5
        L_eff = D_eff[i] / k_MT[i]
        J[i, N_CL+1] = np_flux(c[i, N_CL], c_eq_vec[i], phi_L_cell[N_CL], zero(ElT),
                                D_eff[i], z_liq[i], L_eff)
    end

    # Cs+ migration flux -- not a DOF (recovered from electroneutrality), but its current MUST
    # enter i_L (§3.1, §9.4 item 8; BUILD_DIALOGUE item 4). No flux at either CL boundary.
    cCs = [c_Cs_from_electroneutrality(c[1,ix], c[2,ix], c[3,ix], c[4,ix]) for ix in 1:N_CL]
    J_Cs = zeros(ElT, N_CL+1)
    for f in 2:N_CL
        J_Cs[f] = np_flux(cCs[f-1], cCs[f], phi_L_cell[f-1], phi_L_cell[f],
                           D_eff_Cs, z_Cs, cl_mesh.dx_between[f])
    end

    # Step 2: faradaic rates + phase transfer, using LOCAL phi_s, phi_L
    R_CO2R_cell = zeros(ElT, N_CL); R_HER_cell = zeros(ElT, N_CL); R_PT_cell = zeros(ElT, N_CL)
    for ix in 1:N_CL
        c_ix = view(c, :, ix)
        phi_L_local = phi_L_cell[ix]; phi_s_local = phi_s_cl[ix]
        R_CO2R_cell[ix] = R_CO2R(c_ix[5], phi_s_local, phi_L_local, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e_val)
        R_HER_cell[ix]  = R_HER(phi_s_local, phi_L_local, c_ix[1], a_v_CL, n_e_val)
        p_CO2_local = mole_fraction_CO2(omega_cl[ix]) * p_cl[ix] / 101325.0
        R_PT_cell[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(c_ix) * p_CO2_local - c_ix[5])
    end

    # Step 3: liquid species residuals
    for ix in 1:N_CL
        buf = buffer_sources(view(c, :, ix))
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix])
            if i == 5
                src -= R_CO2R_cell[ix]
                src += R_PT_cell[ix]
            end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end

    # Step 4: gas-phase residual (Stage 3-4 physics, convective=true)
    gas_residual!(F, ly, omega_gdl, p_gdl, omega_cl, p_cl, R_PT_cell, R_CO2R_cell, R_HER_cell,
                  gdl_mesh, cl_mesh; convective=true)

    # Step 5: phi_L residual -- local ionic charge balance per CL cell, Cs+ included in i_L_face
    i_L_face = zeros(ElT, N_CL+1)
    for f in 1:N_CL+1
        i_L_face[f] = F_const * (sum(z_liq[i] * J[i, f] for i in 1:5) + z_Cs * J_Cs[f])
    end
    for ix in 1:N_CL
        i_far_vol = n_e_val * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_L_idx5(ly, ix)] = (i_L_face[ix+1] - i_L_face[ix]) + i_far_vol * cl_mesh.dx[ix]
    end
    F[phi_L_idx5(ly, N_CL)] = phi_L_cell[N_CL]     # Dirichlet gauge override (§3.1)

    # Step 6: phi_s residual on GDL + CL
    i_s = solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, sigma_s_eff_GDL, sigma_s_eff_CL,
                               gdl_mesh, cl_mesh)
    for ig in 1:N_GDL
        F[phi_s_gdl_idx5(ly, ig)] = i_s[ig+1] - i_s[ig]
    end
    for ix in 1:N_CL
        i_far_vol = n_e_val * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
        F[phi_s_cl_idx5(ly, ix)] = (i_s[N_GDL+1+ix] - i_s[N_GDL+ix]) - i_far_vol * cl_mesh.dx[ix]
    end
    return F
end

# ── Current extractors (§17.4). `phi_s_of`/`phi_L_of` are per-cell closures: constants for
# Stages 1-4 (`ix -> V_applied_SHE`, `ix -> 0.0`), DOF-reading for Stage 5. ──

function compute_i_CO(u::AbstractVector{<:Real}, cl_mesh::FVMesh, phi_s_of::Function, phi_L_of::Function)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    i = 0.0
    for ix in 1:N
        r = R_CO2R(c[5,ix], phi_s_of(ix), phi_L_of(ix), c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e_val)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
function compute_i_H2(u::AbstractVector{<:Real}, cl_mesh::FVMesh, phi_s_of::Function, phi_L_of::Function)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    i = 0.0
    for ix in 1:N
        r = R_HER(phi_s_of(ix), phi_L_of(ix), c[1,ix], a_v_CL, n_e_val)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
compute_i_total(u, cl_mesh, phi_s_of, phi_L_of) =
    compute_i_CO(u, cl_mesh, phi_s_of, phi_L_of) + compute_i_H2(u, cl_mesh, phi_s_of, phi_L_of)

"Convenience for Stages 1-4: constant phi_s=V_applied_SHE, phi_L=0 everywhere."
const_phi_closures(V_applied_SHE::Real) = (ix -> V_applied_SHE), (ix -> 0.0)
