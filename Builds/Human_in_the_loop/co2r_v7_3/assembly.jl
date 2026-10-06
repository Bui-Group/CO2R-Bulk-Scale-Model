# ============================================================================
# assembly.jl — the monolithic residual for every stage, plus current
#               extractors and the per-stage initial-guess builders.
#
# Stage switches (§11):
#   1  liquid diffusion only, Dirichlet at x = L_CL, van 't Hoff Henry BC
#   2  + Sherwood-Reynolds Neumann at x = L_CL
#   3  + Stefan-Maxwell gas (u_G = 0, p pinned) + R_PT + Sechenov Henry
#   4  + Darcy pressure equation and convective gas flux
#   5  + φ_L (NP migration, monolithic DOF) and φ_s (solid Ohm's law)
# Stages 1–4 use the simplified kinetics φ_s = V_applied_SHE, φ_L = 0.
# ============================================================================

struct Ctx
    ly::Layout
    cl::Mesh1D
    gdl::Mesh1D
    gm::GasMesh
    c_eq::Vector{Float64}
end

function Ctx(stage::Int, N_CL::Int, N_GDL::Int, c_eq::Vector{Float64})
    cl  = make_cl_mesh(N_CL, L_CL)
    gdl = make_gdl_mesh(N_GDL, L_GDL)
    gm  = GasMesh(gdl, cl)
    Ctx(Layout(stage, N_CL, N_GDL), cl, gdl, gm, c_eq)
end

const P_CO2_INLET_ATM = x_inlet[1] * p_total / 101325.0     # 0.88 atm

# ────────────────────────────────────────────────────────────────────────────
function residual!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                   V_cathode::Real, ctx::Ctx) where {ElT<:Real}
    ly, cl, gdl, gm, c_eq = ctx.ly, ctx.cl, ctx.gdl, ctx.gm, ctx.c_eq
    N     = ly.N_CL
    stage = ly.stage
    has_gas  = stage >= 3
    darcy_on = stage >= 4
    has_pot  = stage >= 5
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    c = unpack_log_conc(u, ly)

    # local potentials at each CL cell (§11: Stages 1–4 pass constants)
    phi_s_cell = zeros(ElT, N)
    phi_L_cell = zeros(ElT, N)
    if has_pot
        @inbounds for ix in 1:N
            phi_s_cell[ix] = u[phi_s_cl_idx(ly, ix)]
            phi_L_cell[ix] = u[phi_L_idx(ly, ix)]
        end
    else
        @inbounds for ix in 1:N
            phi_s_cell[ix] = ElT(V_applied_SHE)
        end
    end

    # ── Step 1: liquid face fluxes ─────────────────────────────────────────
    J = zeros(ElT, 5, N + 1)

    # left (GDL/CL) face: ions no-flux, CO2(aq) from Henry's law
    c_CO2_bc = if has_gas
        w1  = gas_omega(u, ly, gm.N_GDL + 1)
        x1  = omega_to_x(w1)
        p1  = gas_p(u, ly, gm.N_GDL + 1)
        p_CO2_local = x1[1] * p1 / 101325.0                 # atm
        sechenov_H_CO2(view(c, :, 1)) * p_CO2_local         # §7.3 / §7.4
    else
        ElT(H_vantHoff * P_CO2_INLET_ATM)                   # §7.1  ≈ 16.0 mol/m³
    end
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5, 1] = has_pot ? zero(ElT) :
              diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], cl.dx_between[1])

    @inbounds for f in 2:N, i in 1:5
        J[i, f] = has_pot ?
            np_flux(c[i, f-1], c[i, f], phi_L_cell[f-1], phi_L_cell[f],
                    D_eff[i], z_sp[i], cl.dx_between[f]) :
            diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl.dx_between[f])
    end

    if stage == 1                                            # §7.1 Dirichlet
        @inbounds for i in 1:5
            J[i, N+1] = diffusion_flux(c[i, N], c_eq[i], D_eff[i], cl.dx_between[N+1])
        end
    elseif has_pot                                           # §11 Stage 5 #12
        @inbounds for i in 1:5
            L_eff = D_eff[i] / k_MT[i]
            J[i, N+1] = np_flux(c[i, N], c_eq[i], phi_L_cell[N], zero(ElT),
                                D_eff[i], z_sp[i], L_eff)
        end
    else                                                     # §7.2 Sherwood
        @inbounds for i in 1:5
            J[i, N+1] = -k_MT[i] * (c_eq[i] - c[i, N])
        end
    end

    # ── Step 2: faradaic rates and phase transfer per CL cell ──────────────
    R_co2r = zeros(ElT, N); R_her = zeros(ElT, N); R_pt = zeros(ElT, N)
    @inbounds for ix in 1:N
        cv = view(c, :, ix)
        R_co2r[ix] = R_CO2R(cv[5], phi_s_cell[ix], phi_L_cell[ix], cv[1],
                            a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        R_her[ix]  = R_HER(phi_s_cell[ix], phi_L_cell[ix], cv[1], a_v_CL, n_e)
        if has_gas
            g  = gm.N_GDL + ix
            wg = gas_omega(u, ly, g)
            xg = omega_to_x(wg)
            pg = gas_p(u, ly, g)
            p_CO2_local = xg[1] * pg / 101325.0
            R_pt[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(cv) * p_CO2_local - cv[5])
        end
    end

    # ── Step 3: liquid species residuals (integrated FV, §3.2/§6.4) ────────
    @inbounds for ix in 1:N
        buf = buffer_sources(view(c, :, ix))
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2
                src += 2*R_co2r[ix] + 2*R_her[ix]
            elseif i == 5
                src -= R_co2r[ix]
                has_gas && (src += R_pt[ix])
            end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl.dx[ix]
        end
    end

    # ── Step 4: gas-phase residual (§6) ────────────────────────────────────
    if has_gas
        NG = gm.N
        Jg   = zeros(ElT, 4, NG + 1)
        rhou = zeros(ElT, NG + 1)

        w_in = (ElT(omega_inlet[1]), ElT(omega_inlet[2]),
                ElT(omega_inlet[3]), ElT(omega_inlet[4]))
        j1, j2, j3, j4, ru = sm_face_flux(w_in, ElT(p_total),
                                          gas_omega(u, ly, 1), gas_p(u, ly, 1),
                                          gm.dx_between[1], face_Dbin(gm, 1),
                                          face_kappa(gm, 1), darcy_on)
        Jg[1,1] = j1; Jg[2,1] = j2; Jg[3,1] = j3; Jg[4,1] = j4; rhou[1] = ru

        @inbounds for f in 2:NG
            j1, j2, j3, j4, ru = sm_face_flux(gas_omega(u, ly, f-1), gas_p(u, ly, f-1),
                                              gas_omega(u, ly, f),   gas_p(u, ly, f),
                                              gm.dx_between[f], face_Dbin(gm, f),
                                              face_kappa(gm, f), darcy_on)
            Jg[1,f] = j1; Jg[2,f] = j2; Jg[3,f] = j3; Jg[4,f] = j4; rhou[f] = ru
        end
        # right (electrolyte-side) face: zero gas flux, dp/dx = 0  (§7.3)

        @inbounds for g in 1:NG
            S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
            if g > gm.N_GDL
                ix = g - gm.N_GDL
                S1 = -R_pt[ix]   * M_gas[1]
                S2 =  R_co2r[ix] * M_gas[2]
                S3 =  R_her[ix]  * M_gas[3]
            end
            F[gas_idx(ly, g, 1)] = Jg[1,g] - Jg[1,g+1] + S1 * gm.dx[g]
            F[gas_idx(ly, g, 2)] = Jg[2,g] - Jg[2,g+1] + S2 * gm.dx[g]
            F[gas_idx(ly, g, 3)] = Jg[3,g] - Jg[3,g+1] + S3 * gm.dx[g]
            if darcy_on
                F[gas_idx(ly, g, 4)] = rhou[g] - rhou[g+1] + (S1 + S2 + S3) * gm.dx[g]
            else
                F[gas_idx(ly, g, 4)] = gas_p(u, ly, g) - p_total      # §11 Stage 3 pin
            end
        end
    end

    # ── Step 5/6: potentials (Stage 5 only) ────────────────────────────────
    if has_pot
        # Cs⁺ from electroneutrality; its migration current MUST enter i_L (§9.4 #8)
        cCs = zeros(ElT, N)
        @inbounds for ix in 1:N
            cCs[ix] = c_Cs_from_electroneutrality(view(c, :, ix))
        end
        J_Cs = zeros(ElT, N + 1)          # J_Cs[1] = J_Cs[N+1] = 0 (§3.1)
        @inbounds for f in 2:N
            J_Cs[f] = np_flux(cCs[f-1], cCs[f], phi_L_cell[f-1], phi_L_cell[f],
                              D_eff_Cs, z_Cs, cl.dx_between[f])
        end

        i_L_face = zeros(ElT, N + 1)
        @inbounds for f in 1:N+1
            s = zero(ElT)
            for i in 1:5
                s += z_sp[i] * J[i, f]
            end
            i_L_face[f] = F_const * (s + z_Cs * J_Cs[f])
        end

        @inbounds for ix in 1:N
            i_far_vol = n_e * F_const * (R_co2r[ix] + R_her[ix])
            F[phi_L_idx(ly, ix)] = (i_L_face[ix+1] - i_L_face[ix]) + i_far_vol * cl.dx[ix]
        end
        F[phi_L_idx(ly, N)] = phi_L_cell[N]        # Dirichlet gauge override (§3.1)

        # φ_s: solid Ohm's law on GDL + CL (§3.4), ghost-cell Dirichlet at x = -L_GDL
        NG = gm.N
        i_s = zeros(ElT, NG + 1)
        i_s[1] = -sigma_s_eff_GDL *
                 (u[phi_s_gdl_idx(ly, 1)] - V_applied_SHE) / gm.dx_between[1]
        @inbounds for f in 2:NG
            sig = f <= gm.N_GDL ? sigma_s_eff_GDL :
                  (f == gm.N_GDL + 1 ? sigma_s_eff_INT : sigma_s_eff_CL)
            pl = (f-1) <= gm.N_GDL ? u[phi_s_gdl_idx(ly, f-1)] :
                                     u[phi_s_cl_idx(ly, f-1-gm.N_GDL)]
            pr = f <= gm.N_GDL ? u[phi_s_gdl_idx(ly, f)] :
                                 u[phi_s_cl_idx(ly, f-gm.N_GDL)]
            i_s[f] = -sig * (pr - pl) / gm.dx_between[f]
        end
        # i_s[NG+1] = 0 : Neumann at the electrolyte side

        @inbounds for ig in 1:gm.N_GDL
            F[phi_s_gdl_idx(ly, ig)] = i_s[ig+1] - i_s[ig]
        end
        @inbounds for ix in 1:N
            i_far_vol = n_e * F_const * (R_co2r[ix] + R_her[ix])
            F[phi_s_cl_idx(ly, ix)] =
                (i_s[gm.N_GDL+1+ix] - i_s[gm.N_GDL+ix]) - i_far_vol * cl.dx[ix]
        end
    end

    return F
end

# ────────────────────────────────────────────────────────────────────────────
# Current extractors (§17.4) — A/m² of geometric CL cross-section
# ────────────────────────────────────────────────────────────────────────────
function cell_rates(u::Vector{Float64}, V_cathode::Float64, ctx::Ctx)
    ly, cl = ctx.ly, ctx.cl
    N = ly.N_CL
    c = unpack_log_conc(u, ly)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    rco = zeros(N); rh = zeros(N)
    for ix in 1:N
        ps = ly.stage >= 5 ? u[phi_s_cl_idx(ly, ix)] : V_applied_SHE
        pl = ly.stage >= 5 ? u[phi_L_idx(ly, ix)]    : 0.0
        rco[ix] = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        rh[ix]  = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
    end
    return rco, rh
end

function compute_currents(u::Vector{Float64}, V_cathode::Float64, ctx::Ctx)
    rco, rh = cell_rates(u, V_cathode, ctx)
    dx = ctx.cl.dx
    i_CO = n_e * F_const * sum(rco .* dx)
    i_H2 = n_e * F_const * sum(rh  .* dx)
    return (i_CO, i_H2, i_CO + i_H2)
end

# ────────────────────────────────────────────────────────────────────────────
# Initial-guess builders (§9.5.5 anchor recipes / §9.6)
# ────────────────────────────────────────────────────────────────────────────
function make_u0(ctx::Ctx)
    ly = ctx.ly
    u = zeros(ly.N_TOT)
    lnc = log.(ctx.c_eq)
    for ix in 1:ly.N_CL, i in 1:5
        u[liquid_idx(ix, i)] = lnc[i]
    end
    if ly.stage >= 3
        for g in 1:(ly.N_GDL + ly.N_CL)
            u[gas_idx(ly, g, 1)] = omega_inlet[1]
            u[gas_idx(ly, g, 2)] = omega_inlet[2]
            u[gas_idx(ly, g, 3)] = omega_inlet[3]
            u[gas_idx(ly, g, 4)] = p_total
        end
    end
    if ly.stage >= 5
        # §11 Stage 5 #9a: write V_applied_SHE (NOT V_cathode) into φ_s.
        # The anchor is V_c = 0 ⇒ V_applied_SHE = 0 ⇒ φ_s = 0, φ_L = 0.
        for ig in 1:ly.N_GDL; u[phi_s_gdl_idx(ly, ig)] = 0.0; end
        for ix in 1:ly.N_CL;  u[phi_s_cl_idx(ly, ix)]  = 0.0; end
        for ix in 1:ly.N_CL;  u[phi_L_idx(ly, ix)]     = 0.0; end
    end
    return u
end
