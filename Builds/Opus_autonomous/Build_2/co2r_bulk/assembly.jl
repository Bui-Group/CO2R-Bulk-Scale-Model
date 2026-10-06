# assembly.jl — DOF layout, index functions, the unified residual, current extractors
#
# DOF layout (§2 / §11 Stage 5):
#   u = [ ln(c)_1..5N_CL ; gas_1..4(N_GDL+N_CL) ;
#         φ_s_GDL_1..N_GDL ; φ_s_CL_1..N_CL ; φ_L_CL_1..N_CL ]
# Stages 1–2 stop after the liquid block, Stages 3–4 after the gas block.

struct Layout
    N_CL::Int
    N_GDL::Int
    stage::Int
    N_TOT::Int
end

function Layout(N_CL::Int, N_GDL::Int, stage::Int)
    n = 5 * N_CL
    stage >= 3 && (n += 4 * (N_GDL + N_CL))
    stage >= 5 && (n += N_GDL + N_CL + N_CL)
    return Layout(N_CL, N_GDL, stage, n)
end

@inline liquid_idx(ly::Layout, ix::Int, i::Int) = 5*(ix - 1) + i
@inline _gas_start(ly::Layout) = 5 * ly.N_CL
# combined gas cell index k = 1..N_GDL (GDL), N_GDL+1..N_GDL+N_CL (CL)
@inline gas_idx(ly::Layout, k::Int, j::Int) = _gas_start(ly) + 4*(k - 1) + j
@inline gas_gdl_idx(ly::Layout, ig::Int, j::Int) = gas_idx(ly, ig, j)
@inline gas_cl_idx(ly::Layout, ix::Int, j::Int)  = gas_idx(ly, ly.N_GDL + ix, j)
@inline _phi_s_start(ly::Layout) = _gas_start(ly) + 4*(ly.N_GDL + ly.N_CL)
@inline phi_s_gdl_idx(ly::Layout, ig::Int) = _phi_s_start(ly) + ig
@inline phi_s_cl_idx(ly::Layout, ix::Int)  = _phi_s_start(ly) + ly.N_GDL + ix
@inline phi_L_idx(ly::Layout, ix::Int)     = _phi_s_start(ly) + ly.N_GDL + ly.N_CL + ix

"DOF blocks used to build the affine-covariant weight vector (§9.5.3 step 1)"
function dof_blocks(ly::Layout)
    b = Vector{UnitRange{Int}}()
    push!(b, 1:(5*ly.N_CL))                                        # liquid log-c
    if ly.stage >= 3
        g0 = _gas_start(ly)
        push!(b, (g0+1):(g0 + 4*ly.N_GDL))                         # gas GDL
        push!(b, (g0 + 4*ly.N_GDL + 1):(g0 + 4*(ly.N_GDL+ly.N_CL)))# gas CL
    end
    if ly.stage >= 5
        p0 = _phi_s_start(ly)
        push!(b, (p0+1):(p0+ly.N_GDL))                             # φ_s GDL
        push!(b, (p0+ly.N_GDL+1):(p0+ly.N_GDL+ly.N_CL))            # φ_s CL
        push!(b, (p0+ly.N_GDL+ly.N_CL+1):ly.N_TOT)                 # φ_L
    end
    return b
end

# ── unpacking helpers (element type must follow u for ForwardDiff) ─────────
function unpack_log_conc(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    c = zeros(ElT, 5, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ly, ix, i)], -50.0, 50.0))
    end
    return c
end

@inline function gas_omega(u::AbstractVector{ElT}, ly::Layout, k::Int) where {ElT<:Real}
    w1 = u[gas_idx(ly, k, 1)]; w2 = u[gas_idx(ly, k, 2)]; w3 = u[gas_idx(ly, k, 3)]
    return (w1, w2, w3, omega_H2O_closure(w1, w2, w3))
end
@inline gas_p(u::AbstractVector, ly::Layout, k::Int) = u[gas_idx(ly, k, 4)]

# ── solid-phase face currents (§3.4) ──────────────────────────────────────
"""
    solid_face_currents!(is, u, ly, cl, gdl, V_applied_SHE)

Fill `is` (length N_GDL+N_CL+1) with the electronic face current density
[A/m²].  Left face uses the GHOST-CELL Dirichlet at the current collector;
right face is Neumann zero.  The GDL/CL interface uses the harmonic mean of the
two effective conductivities.
"""
function solid_face_currents!(is::AbstractVector{ElT}, u::AbstractVector{ElT},
                              ly::Layout, cl::Mesh1D, gdl::Mesh1D,
                              V_applied_SHE::Real) where {ElT<:Real}
    NG, NC = ly.N_GDL, ly.N_CL
    @inbounds begin
        # face 1 — ghost cell at the current collector (NOT a residual override)
        is[1] = -sigma_s_eff_GDL * (u[phi_s_gdl_idx(ly,1)] - V_applied_SHE) / gdl.dxc[1]
        for f in 2:NG                                     # GDL interior
            is[f] = -sigma_s_eff_GDL *
                    (u[phi_s_gdl_idx(ly,f)] - u[phi_s_gdl_idx(ly,f-1)]) / gdl.dxc[f]
        end
        # GDL/CL interface — harmonic mean conductivity
        sig_int = 2*sigma_s_eff_GDL*sigma_s_eff_CL / (sigma_s_eff_GDL + sigma_s_eff_CL)
        d_int = gdl.dxc[NG+1] + cl.dxc[1]
        is[NG+1] = -sig_int * (u[phi_s_cl_idx(ly,1)] - u[phi_s_gdl_idx(ly,NG)]) / d_int
        for i in 2:NC                                     # CL interior
            is[NG+i] = -sigma_s_eff_CL *
                       (u[phi_s_cl_idx(ly,i)] - u[phi_s_cl_idx(ly,i-1)]) / cl.dxc[i]
        end
        is[NG+NC+1] = zero(ElT)                           # Neumann at electrolyte side
    end
    return is
end

# ══════════════════════════════════════════════════════════════════════════
#  The unified residual.  Val{S} selects the stage at compile time:
#    S = 1 : liquid diffusion, Dirichlet electrolyte BC
#    S = 2 : + Sherwood-Reynolds Neumann electrolyte BC
#    S = 3 : + Stefan-Maxwell gas (u_G = 0, pressure pinned) + R_PT
#    S = 4 : + Darcy pressure equation and convective gas flux
#    S = 5 : + φ_L (NP migration) and φ_s (solid Ohm's law) as DOFs
# ══════════════════════════════════════════════════════════════════════════
function full_residual!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                        V_cathode::Real, cl::Mesh1D, gdl::Mesh1D,
                        c_eq::Vector{Float64}, ly::Layout, geom::GasGeom,
                        ::Val{S}) where {ElT<:Real, S}
    N  = ly.N_CL
    NG = ly.N_GDL
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, ly)

    has_gas = (S >= 3)
    darcy   = (S >= 4)
    pots    = (S == 5)

    # ── liquid potential (Stage 5 only) ───────────────────────────────────
    phiL = zeros(ElT, N)
    if pots
        @inbounds for ix in 1:N
            phiL[ix] = u[phi_L_idx(ly, ix)]
        end
    end

    # ── Step 1: liquid face fluxes ────────────────────────────────────────
    J = zeros(ElT, 5, N + 1)
    # left face (x = 0): ions no-flux; CO₂(aq) Dirichlet from Henry's law
    if has_gas
        w = gas_omega(u, ly, NG + 1)
        Mn = Mn_from_omega(w[1], w[2], w[3], w[4])
        p_cl1 = gas_p(u, ly, NG + 1)
        x_CO2 = omega_to_x(w[1], Mn, 1)
        p_CO2_atm = x_CO2 * p_cl1 / 101325.0
        H_loc = sechenov_H_CO2(c[1,1], c[2,1], c[3,1], c[4,1])
        c_CO2_bc = H_loc * p_CO2_atm
    else
        c_CO2_bc = c_CO2_henry_s12
    end
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    @inbounds J[5,1] = S >= 5 ? zero(ElT) :
                       diffusion_flux(c_CO2_bc, c[5,1], D_eff[5], cl.dxc[1])

    # interior faces
    @inbounds for f in 2:N, i in 1:5
        J[i,f] = pots ?
            np_flux(c[i,f-1], c[i,f], phiL[f-1], phiL[f], D_eff[i], z_sp[i], cl.dxc[f]) :
            diffusion_flux(c[i,f-1], c[i,f], D_eff[i], cl.dxc[f])
    end

    # right face (x = L_CL)
    @inbounds for i in 1:5
        if S == 1
            J[i,N+1] = diffusion_flux(c[i,N], c_eq[i], D_eff[i], cl.dxc[N+1])
        elseif pots
            L_eff = D_eff[i] / k_MT[i]
            J[i,N+1] = np_flux(c[i,N], c_eq[i], phiL[N], zero(ElT), D_eff[i], z_sp[i], L_eff)
        else
            J[i,N+1] = -k_MT[i] * (c_eq[i] - c[i,N])
        end
    end

    # ── Step 2: faradaic rates and phase transfer per CL cell ─────────────
    Rco = zeros(ElT, N); Rher = zeros(ElT, N); Rpt = zeros(ElT, N)
    @inbounds for ix in 1:N
        phi_s_local = pots ? u[phi_s_cl_idx(ly, ix)] : ElT(V_applied_SHE)
        phi_L_local = pots ? phiL[ix] : zero(ElT)
        Rco[ix]  = R_CO2R(c[5,ix], phi_s_local, phi_L_local, c[1,ix],
                          a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rher[ix] = R_HER(phi_s_local, phi_L_local, c[1,ix], a_v_CL, n_e)
        if has_gas
            w = gas_omega(u, ly, NG + ix)
            Mn = Mn_from_omega(w[1], w[2], w[3], w[4])
            p_atm = omega_to_x(w[1], Mn, 1) * gas_p(u, ly, NG + ix) / 101325.0
            H_loc = sechenov_H_CO2(c[1,ix], c[2,ix], c[3,ix], c[4,ix])
            Rpt[ix] = a_v_GL * k_GL() * (H_loc * p_atm - c[5,ix])
        end
    end

    # ── Step 3: liquid species residuals (integrated FV, §6.4) ────────────
    @inbounds for ix in 1:N
        buf = buffer_sources(c[1,ix], c[2,ix], c[3,ix], c[4,ix], c[5,ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*Rco[ix] + 2*Rher[ix])
            if i == 5
                src -= Rco[ix]
                has_gas && (src += Rpt[ix])
            end
            F[liquid_idx(ly, ix, i)] = J[i,ix] - J[i,ix+1] + src * cl.dx[ix]
        end
    end

    # ── Step 4: gas-phase residual (Stefan-Maxwell + Darcy) ───────────────
    if has_gas
        NK = geom.NK
        Jg  = zeros(ElT, 3, NK + 1)
        rhou = zeros(ElT, NK + 1)
        # face 1: inlet ghost cell at ω_inlet, p_total
        wgh = (ElT(omega_inlet[1]), ElT(omega_inlet[2]), ElT(omega_inlet[3]), ElT(omega_inlet[4]))
        w1 = gas_omega(u, ly, 1)
        (a, b, d, ru) = sm_face_flux_full(wgh, w1, ElT(p_total), gas_p(u, ly, 1),
                                          geom, 1, darcy)
        Jg[1,1] = a; Jg[2,1] = b; Jg[3,1] = d; rhou[1] = ru
        @inbounds for f in 2:NK
            wL = gas_omega(u, ly, f-1); wR = gas_omega(u, ly, f)
            (a, b, d, ru) = sm_face_flux_full(wL, wR, gas_p(u, ly, f-1), gas_p(u, ly, f),
                                              geom, f, darcy)
            Jg[1,f] = a; Jg[2,f] = b; Jg[3,f] = d; rhou[f] = ru
        end
        # right face: no gas flux into the electrolyte, dp/dx = 0
        Jg[1,NK+1] = zero(ElT); Jg[2,NK+1] = zero(ElT); Jg[3,NK+1] = zero(ElT)
        rhou[NK+1] = zero(ElT)

        @inbounds for k in 1:NK
            in_cl = k > NG
            ix = k - NG
            S1 = in_cl ? -Rpt[ix] * M_gas[1] : zero(ElT)
            S2 = in_cl ?  Rco[ix] * M_gas[2] : zero(ElT)
            S3 = in_cl ?  Rher[ix] * M_gas[3] : zero(ElT)
            F[gas_idx(ly,k,1)] = Jg[1,k] - Jg[1,k+1] + S1 * geom.dx[k]
            F[gas_idx(ly,k,2)] = Jg[2,k] - Jg[2,k+1] + S2 * geom.dx[k]
            F[gas_idx(ly,k,3)] = Jg[3,k] - Jg[3,k+1] + S3 * geom.dx[k]
            if darcy
                F[gas_idx(ly,k,4)] = rhou[k] - rhou[k+1] + (S1 + S2 + S3) * geom.dx[k]
            else
                F[gas_idx(ly,k,4)] = gas_p(u, ly, k) - p_total     # Stage 3 pin
            end
        end
    end

    # ── Step 5: φ_L residual — per-cell ionic charge balance (§3.1) ───────
    if pots
        iL = zeros(ElT, N + 1)
        @inbounds for f in 1:(N+1)
            s = zero(ElT)
            for i in 1:5
                s += z_sp[i] * J[i,f]
            end
            # Cs⁺ migration MUST be included (§9.4 item 8).  No Cs⁺ flux at
            # either CL boundary face.
            if f >= 2 && f <= N
                cCsL = cCs_from_electroneutrality(c[1,f-1], c[2,f-1], c[3,f-1], c[4,f-1])
                cCsR = cCs_from_electroneutrality(c[1,f],   c[2,f],   c[3,f],   c[4,f])
                s += z_Cs * np_flux(cCsL, cCsR, phiL[f-1], phiL[f],
                                    D_eff_Cs, z_Cs, cl.dxc[f])
            end
            iL[f] = F_const * s
        end
        @inbounds for ix in 1:N
            i_far_vol = n_e * F_const * (Rco[ix] + Rher[ix])
            F[phi_L_idx(ly, ix)] = (iL[ix+1] - iL[ix]) + i_far_vol * cl.dx[ix]
        end
        # Dirichlet gauge override at the electrolyte-side cell
        F[phi_L_idx(ly, N)] = phiL[N]

        # ── Step 6: φ_s residual on GDL + CL (§3.4) ───────────────────────
        is = zeros(ElT, NG + N + 1)
        solid_face_currents!(is, u, ly, cl, gdl, V_applied_SHE)
        @inbounds for ig in 1:NG
            F[phi_s_gdl_idx(ly, ig)] = is[ig+1] - is[ig]
        end
        @inbounds for ix in 1:N
            i_far_vol = n_e * F_const * (Rco[ix] + Rher[ix])
            F[phi_s_cl_idx(ly, ix)] = (is[NG+1+ix] - is[NG+ix]) - i_far_vol * cl.dx[ix]
        end
    end
    return F
end

# thin per-stage wrappers (the guide's residual_stageN! names)
residual_stage1!(F,u,V,cl,gdl,ceq,ly,geom) = full_residual!(F,u,V,cl,gdl,ceq,ly,geom,Val(1))
residual_stage2!(F,u,V,cl,gdl,ceq,ly,geom) = full_residual!(F,u,V,cl,gdl,ceq,ly,geom,Val(2))
residual_stage3!(F,u,V,cl,gdl,ceq,ly,geom) = full_residual!(F,u,V,cl,gdl,ceq,ly,geom,Val(3))
residual_stage4!(F,u,V,cl,gdl,ceq,ly,geom) = full_residual!(F,u,V,cl,gdl,ceq,ly,geom,Val(4))
residual_stage5!(F,u,V,cl,gdl,ceq,ly,geom) = full_residual!(F,u,V,cl,gdl,ceq,ly,geom,Val(5))

# ══════════════════════════════════════════════════════════════════════════
#  Current extractors (§17.4)
# ══════════════════════════════════════════════════════════════════════════
"""
    compute_rates(u, ly, cl, V_cathode) -> (R_CO2R[1:N], R_HER[1:N])

Volumetric reaction-rate profiles [mol/m³/s].  For Stages 1–4 the kinetics use
(V_applied_SHE, 0); for Stage 5 they read the local φ_s and φ_L DOFs.
"""
function compute_rates(u::Vector{Float64}, ly::Layout, cl::Mesh1D, V_cathode::Float64)
    V_app = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, ly)
    N = ly.N_CL
    Rco = zeros(N); Rher = zeros(N)
    for ix in 1:N
        ps = ly.stage >= 5 ? u[phi_s_cl_idx(ly, ix)] : V_app
        pl = ly.stage >= 5 ? u[phi_L_idx(ly, ix)]    : 0.0
        Rco[ix]  = R_CO2R(c[5,ix], ps, pl, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rher[ix] = R_HER(ps, pl, c[1,ix], a_v_CL, n_e)
    end
    return Rco, Rher
end

"Partial current densities [A/m²] of CL geometric cross-section"
function compute_currents(u::Vector{Float64}, ly::Layout, cl::Mesh1D, V_cathode::Float64)
    Rco, Rher = compute_rates(u, ly, cl, V_cathode)
    iCO = 0.0; iH2 = 0.0
    for ix in 1:ly.N_CL
        iCO += n_e * F_const * Rco[ix] * cl.dx[ix]
        iH2 += n_e * F_const * Rher[ix] * cl.dx[ix]
    end
    return iCO, iH2
end

compute_i_CO(u, ly, cl, V) = compute_currents(u, ly, cl, V)[1]
compute_i_H2(u, ly, cl, V) = compute_currents(u, ly, cl, V)[2]
