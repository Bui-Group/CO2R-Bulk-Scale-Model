# assembly.jl — DOF unpacking, residuals (Stage 1 here), and current extractors.

# Liquid species DOF index: species i (1..5) at CL cell ix.
liquid_idx(ix::Int, i::Int) = 5 * (ix - 1) + i

# §9.5.3 weight blocks for each stage's DOF layout — one block per physical field, so the
# iterate-derived weights w_j = |u_j| + afrac·rms(|u| over the block) see a scale of their own:
#   liquid log-c | gas GDL | gas CL | φ_s GDL | φ_s CL | φ_L   (later stages append blocks)
# Gas index helpers live in gas_transport.jl, potential helpers in assembly_stage5.jl; they are
# only reached for stage ≥ 3 / ≥ 5, when those files are included.
function dof_blocks(stage::Int, N_CL::Int, N_GDL::Int=0)
    blocks = UnitRange{Int}[1:5*N_CL]
    if stage >= 3
        push!(blocks, gas_gdl_idx(1, 1, N_CL):gas_gdl_idx(N_GDL, 4, N_CL))
        push!(blocks, gas_cl_idx(1, 1, N_CL, N_GDL):gas_cl_idx(N_CL, 4, N_CL, N_GDL))
    end
    if stage >= 5
        push!(blocks, phi_s_gdl_idx(1, N_CL, N_GDL):phi_s_gdl_idx(N_GDL, N_CL, N_GDL))
        push!(blocks, phi_s_cl_idx(1, N_CL, N_GDL):phi_s_cl_idx(N_CL, N_CL, N_GDL))
        push!(blocks, phi_L_idx(1, N_CL, N_GDL):phi_L_idx(N_CL, N_CL, N_GDL))
    end
    return blocks
end

"Unpack log-concentration DOFs → 5×N physical concentration matrix (eltype-matched)."
function unpack_log_conc(u::AbstractVector{<:Real}, N::Int)
    ElT = eltype(u)
    c = zeros(ElT, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

"""
    residual_stage1!(F, u, V_cathode, cl_mesh, c_eq) (§Stage 1)

Liquid Nernst-Planck DIFFUSION ONLY (no migration), Dirichlet at both boundaries
(CO₂ via Henry on GDL side, c_eq on electrolyte side), buffer chemistry + CO₂R/HER
kinetics with φ_s = V_applied_SHE and φ_L = 0. Integrated FV form: J_L − J_R + S·dx.
"""
function residual_stage1!(F::AbstractVector{<:Real}, u::AbstractVector{<:Real},
                          V_cathode::Real, cl_mesh, c_eq::AbstractVector)
    ElT = eltype(u)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── Pure diffusion face fluxes (no migration in Stage 1)
    J = zeros(ElT, 5, N + 1)
    J[5, 1] = diffusion_flux(c_CO2_henry, c[5, 1], D_eff[5], cl_mesh.dx_face_left)  # CO₂ Henry BC
    # J[1:4,1] = 0  (ions: no flux at GDL side) — already zero
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl_mesh.dx_cc[f-1])
        end
    end
    for i in 1:5
        J[i, N+1] = diffusion_flux(c[i, N], c_eq[i], D_eff[i], cl_mesh.dx_face_right)  # Dirichlet c_eq
    end

    # ── Cell residuals: integrated FV form
    for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end   # OH⁻ produced by both reactions
            if i == 5; src -= r_co2r;             end   # CO₂(aq) consumed by CO₂R
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end
    return F
end

"""
    residual_stage2!(F, u, V_cathode, cl_mesh, c_eq) (§Stage 2)

Identical to Stage 1 except the electrolyte-side (right) boundary: replace the
Dirichlet diffusion flux with the Sherwood-Reynolds convective Neumann flux.

SIGN CORRECTION (applied 2026-06-18 with user approval — see BUILD_DIALOGUE.md):
The guide's explicit §7.2 line 644 / §Stage-2 line 1721 formula
`J[i,N+1] = k_MT[i]*(c_eq[i]-c[i,N])` has a sign error (it is a destabilizing
wrong-sign Robin BC with no attainable root — see BUILD_DIALOGUE.md and
`test_literal_sign_stage2.jl`). The form used here,
    J[i, N+1] = k_MT[i] * (c[i, N] - c_eq[i])
is the guide's OWN np_flux reduction (§7.2 line 561, §9.1, §11 gotcha #12:
np_flux(c[N], c_eq, φ_L, 0, D_eff, z, L_eff) with L_eff = D_eff/k_MT, at φ_L=0),
is consistent with the diffusion_flux convention, and converges across the sweep.
"""
function residual_stage2!(F::AbstractVector{<:Real}, u::AbstractVector{<:Real},
                          V_cathode::Real, cl_mesh, c_eq::AbstractVector)
    ElT = eltype(u)
    N = cl_mesh.N
    c = unpack_log_conc(u, N)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    J = zeros(ElT, 5, N + 1)
    J[5, 1] = diffusion_flux(c_CO2_henry, c[5, 1], D_eff[5], cl_mesh.dx_face_left)  # CO₂ Henry BC
    # J[1:4,1] = 0  (ions no-flux at GDL side)
    for f in 2:N
        for i in 1:5
            J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl_mesh.dx_cc[f-1])
        end
    end
    # Right face: Sherwood-Reynolds convective Neumann (np_flux-consistent sign)
    for i in 1:5
        J[i, N+1] = k_MT[i] * (c[i, N] - c_eq[i])
    end

    for ix in 1:N
        c_ix = @view c[:, ix]
        r_co2r = R_CO2R(c_ix[5], V_applied_SHE, zero(ElT), c_ix[1],
                        a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), c_ix[1], a_v_CL, n_e)
        buf = buffer_sources(c_ix)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end
            if i == 5; src -= r_co2r;             end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl_mesh.dx[ix]
        end
    end
    return F
end

# ── Current extractors (§17.4). Stage 1–4: pass φ_s = V_applied_SHE, φ_L = 0. ──
"CO partial current density [A/m²] = 2F ∫ R_CO2R dx over the CL."
function compute_i_CO(u::AbstractVector{<:Real}, V_applied_SHE::Real, cl_mesh)
    c = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        r = R_CO2R(c[5, ix], V_applied_SHE, 0.0, c[1, ix],
                   a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end

"H₂ partial current density [A/m²] = 2F ∫ R_HER dx over the CL."
function compute_i_H2(u::AbstractVector{<:Real}, V_applied_SHE::Real, cl_mesh)
    c = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    for ix in 1:cl_mesh.N
        r = R_HER(V_applied_SHE, 0.0, c[1, ix], a_v_CL, n_e)
        i += 2 * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
