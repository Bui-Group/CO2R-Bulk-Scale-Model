# =====================================================================
#  assembly.jl — DOF layout, index maps, full residual, current extractors
#  Stages 1-5 share ONE residual body specialised on Val{S} so the physics
#  cannot drift between stages (§11 stage table).
# =====================================================================

# ── DOF layout (§2, §11 Stage5Layout) ────────────────────────────────
struct ModelLayout
    stage::Int
    N_CL::Int
    N_GDL::Int
    N_TOT::Int
    blocks::Vector{UnitRange{Int}}      # weight blocks for §9.5.3
end

function make_layout(stage::Int, N_CL::Int, N_GDL::Int)
    nliq  = 5 * N_CL
    blocks = UnitRange{Int}[1:nliq]
    ntot  = nliq
    if stage >= 3
        gG = nliq + 1 : nliq + 4*N_GDL
        gC = nliq + 4*N_GDL + 1 : nliq + 4*(N_GDL + N_CL)
        push!(blocks, gG); push!(blocks, gC)
        ntot = nliq + 4*(N_GDL + N_CL)
    end
    if stage == 5
        sG = ntot + 1 : ntot + N_GDL
        sC = ntot + N_GDL + 1 : ntot + N_GDL + N_CL
        pL = ntot + N_GDL + N_CL + 1 : ntot + N_GDL + 2*N_CL
        push!(blocks, sG); push!(blocks, sC); push!(blocks, pL)
        ntot = ntot + N_GDL + 2*N_CL
    end
    return ModelLayout(stage, N_CL, N_GDL, ntot, blocks)
end
Stage5Layout(N_CL::Int, N_GDL::Int) = make_layout(5, N_CL, N_GDL)

# ── index functions (§11) ────────────────────────────────────────────
@inline liquid_idx(ix::Int, i::Int) = 5*(ix - 1) + i
@inline _gas_start(ly)              = 5 * ly.N_CL
@inline gas_gdl_idx5(ly, ig::Int, j::Int) = _gas_start(ly) + 4*(ig - 1) + j
@inline gas_cl_idx5(ly, ix::Int, j::Int)  = _gas_start(ly) + 4*ly.N_GDL + 4*(ix - 1) + j
@inline gas_base(ly, g::Int)              = _gas_start(ly) + 4*(g - 1)   # global gas cell
@inline _phi_s_gdl_start(ly)        = _gas_start(ly) + 4*(ly.N_GDL + ly.N_CL)
@inline phi_s_gdl_idx5(ly, ig::Int) = _phi_s_gdl_start(ly) + ig
@inline phi_s_cl_idx5(ly, ix::Int)  = _phi_s_gdl_start(ly) + ly.N_GDL + ix
@inline phi_L_idx5(ly, ix::Int)     = _phi_s_gdl_start(ly) + ly.N_GDL + ly.N_CL + ix

# ── unpacking helpers (element type matched to `u`, §11 gotcha #1/#3) ─
function unpack_log_conc(u::AbstractVector{ElT}, N::Int) where {ElT<:Real}
    c = zeros(ElT, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end
unpack_log_conc5(u, ly) = unpack_log_conc(u, ly.N_CL)

@inline function gas_omega(u::AbstractVector{ElT}, ly, g::Int) where {ElT<:Real}
    b = gas_base(ly, g)
    w1 = u[b+1]; w2 = u[b+2]; w3 = u[b+3]
    return (w1, w2, w3, omega_H2O_closure(w1, w2, w3))
end
@inline gas_p(u::AbstractVector{ElT}, ly, g::Int) where {ElT<:Real} = u[gas_base(ly, g)+4]
gdl_omega5(u, ly, ig) = gas_omega(u, ly, ig)
gdl_p5(u, ly, ig)     = gas_p(u, ly, ig)
cl_omega5(u, ly, ix)  = gas_omega(u, ly, ly.N_GDL + ix)
cl_p5(u, ly, ix)      = gas_p(u, ly, ly.N_GDL + ix)

function unpack_phi_s_gdl5(u::AbstractVector{ElT}, ly) where {ElT<:Real}
    v = zeros(ElT, ly.N_GDL)
    @inbounds for ig in 1:ly.N_GDL; v[ig] = u[phi_s_gdl_idx5(ly, ig)]; end
    return v
end
function unpack_phi_s_cl5(u::AbstractVector{ElT}, ly) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_s_cl_idx5(ly, ix)]; end
    return v
end
function unpack_phi_L_cl5(u::AbstractVector{ElT}, ly) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_L_idx5(ly, ix)]; end
    return v
end

# ── §3.4 solid-phase face currents on GDL+CL ─────────────────────────
"""
    solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, gdl, cl)

`i_s = −σ_s_eff ∇φ_s` at every face.  Face 1 uses the GHOST-CELL Dirichlet at
the current collector (§3.4 / §11 gotcha #6); the last CL face is Neumann zero.
"""
function solid_face_currents(phi_s_gdl::AbstractVector{ElT}, phi_s_cl::AbstractVector{ElT},
                             V_applied_SHE::Real, gdl::Mesh1D, cl::Mesh1D) where {ElT<:Real}
    Ng = gdl.N; Nc = cl.N
    i_s = zeros(ElT, Ng + Nc + 1)
    i_s[1] = -sigma_s_eff_GDL * (phi_s_gdl[1] - V_applied_SHE) / gdl.dx_between[1]
    @inbounds for k in 2:Ng
        i_s[k] = -sigma_s_eff_GDL * (phi_s_gdl[k] - phi_s_gdl[k-1]) / gdl.dx_between[k]
    end
    d_int = gdl.dx_between[Ng+1] + cl.dx_between[1]
    i_s[Ng+1] = -sigma_s_eff_int * (phi_s_cl[1] - phi_s_gdl[Ng]) / d_int
    @inbounds for m in 2:Nc
        i_s[Ng+m] = -sigma_s_eff_CL * (phi_s_cl[m] - phi_s_cl[m-1]) / cl.dx_between[m]
    end
    i_s[Ng+Nc+1] = zero(ElT)                     # Neumann at the electrolyte side
    return i_s
end

# ── the residual ─────────────────────────────────────────────────────
"""
    full_residual!(F, u, V_cathode, cl, gdl, c_eq, ly, gg, Val(S))

Stage-`S` residual.  `V_cathode` is in **V vs RHE@pH 8** and is converted
internally (§11 gotcha #9a).

  S = 1  liquid diffusion, Dirichlet electrolyte BC, φ_s = V_applied, φ_L = 0
  S = 2  + Sherwood-Reynolds Neumann electrolyte BC
  S = 3  + Stefan-Maxwell gas (u_G = 0) + volumetric phase transfer R_PT
  S = 4  + Darcy pressure equation and convective gas flux
  S = 5  + φ_L (NP migration, charge balance) and φ_s (solid Ohm's law)
"""
function full_residual!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                        V_cathode::Real, cl::Mesh1D, gdl::Mesh1D,
                        c_eq::Vector{Float64}, ly::ModelLayout, gg::GasGeom,
                        ::Val{S}) where {ElT<:Real, S}

    N_CL = cl.N
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N_CL)

    # ── potentials: DOFs in Stage 5, constants otherwise (§11 stage table)
    phi_s_cl = zeros(ElT, N_CL)
    phi_L    = zeros(ElT, N_CL)
    if S == 5
        phi_s_gdl = unpack_phi_s_gdl5(u, ly)
        @inbounds for ix in 1:N_CL
            phi_s_cl[ix] = u[phi_s_cl_idx5(ly, ix)]
            phi_L[ix]    = u[phi_L_idx5(ly, ix)]
        end
    else
        phi_s_gdl = zeros(ElT, 0)
        @inbounds for ix in 1:N_CL; phi_s_cl[ix] = V_applied_SHE; end
    end

    # ── Step 1: liquid face fluxes, integrated FV form ────────────────
    J = zeros(ElT, 5, N_CL + 1)

    # left face (GDL/CL): ions no-flux; CO₂(aq) from Henry's law
    c_CO2_bc = if S <= 2
        ElT(c_CO2_henry_stage12)                                   # §7.1
    else
        w = cl_omega5(u, ly, 1); pcl = cl_p5(u, ly, 1)
        xC = w[1] * Mn_from_omega(w) / M_gas[1]
        sechenov_H_CO2(view(c, :, 1)) * (xC * pcl / 101325.0)      # §7.3 / §7.4
    end
    @inbounds for i in 1:4; J[i, 1] = zero(ElT); end
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5, 1] = S >= 5 ? zero(ElT) :
              diffusion_flux(c_CO2_bc, c[5,1], D_eff[5], cl.dx_between[1])

    # interior faces
    @inbounds for f in 2:N_CL, i in 1:5
        if S == 5
            J[i,f] = np_flux(c[i,f-1], c[i,f], phi_L[f-1], phi_L[f],
                             D_eff[i], z_ion[i], cl.dx_between[f])
        else
            J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], cl.dx_between[f])
        end
    end

    # right face (electrolyte)
    @inbounds for i in 1:5
        if S == 1                                                   # Dirichlet at c_eq
            J[i,N_CL+1] = diffusion_flux(c[i,N_CL], c_eq[i], D_eff[i], cl.dx_between[N_CL+1])
        elseif S <= 4                                               # Sherwood-Reynolds (§7.2)
            J[i,N_CL+1] = diffusion_flux(c[i,N_CL], c_eq[i], D_eff[i], L_eff_bc[i])
        else                                                        # NP incl. migration (§11 #12)
            J[i,N_CL+1] = np_flux(c[i,N_CL], c_eq[i], phi_L[N_CL], zero(ElT),
                                  D_eff[i], z_ion[i], L_eff_bc[i])
        end
    end

    # ── Step 2: faradaic rates and phase transfer per CL cell ─────────
    R_CO2R_cell = zeros(ElT, N_CL)
    R_HER_cell  = zeros(ElT, N_CL)
    R_PT_cell   = zeros(ElT, N_CL)
    @inbounds for ix in 1:N_CL
        R_CO2R_cell[ix] = R_CO2R(c[5,ix], phi_s_cl[ix], phi_L[ix], c[1,ix],
                                 a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_HER_cell[ix]  = R_HER(phi_s_cl[ix], phi_L[ix], c[1,ix], a_v_CL, n_e)
        if S >= 3
            w  = cl_omega5(u, ly, ix); pcl = cl_p5(u, ly, ix)
            xC = w[1] * Mn_from_omega(w) / M_gas[1]
            p_CO2_local = xC * pcl / 101325.0                       # atm
            H_local     = sechenov_H_CO2(view(c, :, ix))
            R_PT_cell[ix] = a_v_GL * k_GL() * (H_local * p_CO2_local - c[5,ix])
        end
    end

    # ── Step 3: liquid species residuals (J_left − J_right + S×dx) ────
    @inbounds for ix in 1:N_CL
        buf = buffer_sources(view(c, :, ix))
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*R_CO2R_cell[ix] + 2*R_HER_cell[ix])
            if i == 5
                src -= R_CO2R_cell[ix]
                S >= 3 && (src += R_PT_cell[ix])
            end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * cl.dx[ix]
        end
    end

    # ── Step 4: gas-phase residual (§6) ───────────────────────────────
    if S >= 3
        Ng = gg.Ng
        darcy = (S >= 4)
        Jg = zeros(ElT, 4, Ng + 1)          # rows 1-3 species mass flux, row 4 = ρu

        # face 1 — flow-field Dirichlet ghost (ω_inlet, p_total)
        wgh = (ElT(omega_inlet[1]), ElT(omega_inlet[2]), ElT(omega_inlet[3]), ElT(omega_inlet[4]))
        wR  = gas_omega(u, ly, 1); pR = gas_p(u, ly, 1)
        j1, j2, j3, ru = sm_face_flux_full(wgh, ElT(p_total), wR, pR,
                                           gg.Dbin[1], gg.kappa[1], gg.dcc[1], darcy)
        Jg[1,1] = j1; Jg[2,1] = j2; Jg[3,1] = j3; Jg[4,1] = ru

        @inbounds for k in 2:Ng
            wL = gas_omega(u, ly, k-1); pL = gas_p(u, ly, k-1)
            wRk= gas_omega(u, ly, k);   pRk= gas_p(u, ly, k)
            a1, a2, a3, aru = sm_face_flux_full(wL, pL, wRk, pRk,
                                                gg.Dbin[k], gg.kappa[k], gg.dcc[k], darcy)
            Jg[1,k] = a1; Jg[2,k] = a2; Jg[3,k] = a3; Jg[4,k] = aru
        end
        # face Ng+1 — electrolyte side: zero species flux, dp/dx = 0 ⇒ ρu = 0
        @inbounds for j in 1:4; Jg[j,Ng+1] = zero(ElT); end

        @inbounds for g in 1:Ng
            s1 = zero(ElT); s2 = zero(ElT); s3 = zero(ElT)
            if g > gg.N_GDL                                          # CL cell sources (§6.5)
                ix = g - gg.N_GDL
                s1 = -R_PT_cell[ix]  * M_gas[1]
                s2 =  R_CO2R_cell[ix]* M_gas[2]
                s3 =  R_HER_cell[ix] * M_gas[3]
            end
            b = gas_base(ly, g)
            F[b+1] = Jg[1,g] - Jg[1,g+1] + s1 * gg.dx[g]
            F[b+2] = Jg[2,g] - Jg[2,g+1] + s2 * gg.dx[g]
            F[b+3] = Jg[3,g] - Jg[3,g+1] + s3 * gg.dx[g]
            if S >= 4                                                # Darcy continuity (§6.4)
                F[b+4] = Jg[4,g] - Jg[4,g+1] + (s1 + s2 + s3) * gg.dx[g]
            else                                                     # Stage 3 pressure placeholder
                F[b+4] = gas_p(u, ly, g) - p_total
            end
        end
    end

    # ── Step 5: φ_L residual — local ionic charge balance (§3.1) ──────
    if S == 5
        # Cs⁺ NP migration MUST enter i_L (§9.4 item 8); no flux at either CL face
        J_Cs = zeros(ElT, N_CL + 1)
        @inbounds for f in 2:N_CL
            cCsL = cCs_from_electroneutrality(view(c, :, f-1))
            cCsR = cCs_from_electroneutrality(view(c, :, f))
            J_Cs[f] = np_flux(cCsL, cCsR, phi_L[f-1], phi_L[f],
                              D_eff_Cs, z_Cs, cl.dx_between[f])
        end

        i_L_face = zeros(ElT, N_CL + 1)
        @inbounds for f in 1:N_CL+1
            acc = zero(ElT)
            for i in 1:5; acc += z_ion[i] * J[i,f]; end
            i_L_face[f] = F_const * (acc + z_Cs * J_Cs[f])
        end

        @inbounds for ix in 1:N_CL
            i_far_vol = n_e * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
            F[phi_L_idx5(ly, ix)] = (i_L_face[ix+1] - i_L_face[ix]) + i_far_vol * cl.dx[ix]
        end
        F[phi_L_idx5(ly, N_CL)] = phi_L[N_CL]        # Dirichlet gauge override (§3.1)

        # ── Step 6: φ_s residual on GDL + CL (§3.4) ───────────────────
        i_s = solid_face_currents(phi_s_gdl, phi_s_cl, V_applied_SHE, gdl, cl)
        Ngd = gdl.N
        @inbounds for ig in 1:Ngd
            F[phi_s_gdl_idx5(ly, ig)] = i_s[ig+1] - i_s[ig]          # no faradaic source
        end
        @inbounds for ix in 1:N_CL
            i_far_vol = n_e * F_const * (R_CO2R_cell[ix] + R_HER_cell[ix])
            F[phi_s_cl_idx5(ly, ix)] = (i_s[Ngd+1+ix] - i_s[Ngd+ix]) - i_far_vol * cl.dx[ix]
        end
        # NOTE: cell-1 φ_s residual is NOT overridden — V_applied_SHE enters
        # through the ghost-cell face current i_s[1] (§3.4, §11 gotcha #6).
    end
    return F
end

# guide-named thin wrappers (§13 canonical entry points)
residual_stage1!(F,u,V,cl,gdl,c_eq,ly,gg) = full_residual!(F,u,V,cl,gdl,c_eq,ly,gg,Val(1))
residual_stage2!(F,u,V,cl,gdl,c_eq,ly,gg) = full_residual!(F,u,V,cl,gdl,c_eq,ly,gg,Val(2))
residual_stage3!(F,u,V,cl,gdl,c_eq,ly,gg) = full_residual!(F,u,V,cl,gdl,c_eq,ly,gg,Val(3))
residual_stage4!(F,u,V,cl,gdl,c_eq,ly,gg) = full_residual!(F,u,V,cl,gdl,c_eq,ly,gg,Val(4))
residual_stage5!(F,u,V,cl,gdl,c_eq,ly,gg) = full_residual!(F,u,V,cl,gdl,c_eq,ly,gg,Val(5))

# ── §17.4 current extractors ─────────────────────────────────────────
"""
    compute_currents(u, V_cathode, cl, ly) -> (i_CO, i_H2, i_tot)  [A/m²]

Partial geometric current densities.  Stages 1-4 use φ_s = V_applied_SHE and
φ_L = 0; Stage 5 reads the φ_s / φ_L DOFs.
"""
function compute_currents(u::Vector{Float64}, V_cathode::Float64, cl::Mesh1D, ly::ModelLayout)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, cl.N)
    iCO = 0.0; iH2 = 0.0
    for ix in 1:cl.N
        ps = ly.stage == 5 ? u[phi_s_cl_idx5(ly, ix)] : V_applied_SHE
        pl = ly.stage == 5 ? u[phi_L_idx5(ly, ix)]    : 0.0
        r1 = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r2 = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
        iCO += n_e * F_const * r1 * cl.dx[ix]
        iH2 += n_e * F_const * r2 * cl.dx[ix]
    end
    return (iCO, iH2, iCO + iH2)
end
compute_i_CO(u, V, cl, ly) = compute_currents(u, V, cl, ly)[1]
compute_i_H2(u, V, cl, ly) = compute_currents(u, V, cl, ly)[2]
