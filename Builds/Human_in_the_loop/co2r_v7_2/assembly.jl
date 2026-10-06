# =============================================================================
# assembly.jl — DOF layout, unpacking helpers, the unified residual for
# Stages 1-5, and the current extractors (§3, §6, §7, §11, §17.4).
#
# One residual function is used for every stage; the stage number in the
# Layout selects the boundary conditions and which physics blocks are active.
# Writing it once (rather than five diverging copies) is what §13 asks for.
# =============================================================================

"""
    Layout(stage, N_CL, N_GDL)

DOF layout (§2, §11 Stage 5):

    u = [ ln(c)_1..5N_CL ; gas_1..4(N_GDL+N_CL) ;
          phi_s_GDL_1..N_GDL ; phi_s_CL_1..N_CL ; phi_L_CL_1..N_CL ]

`blocks` are the natural per-physics DOF blocks used by the affine-covariant
weight construction (§9.5.3): liquid log-c, gas GDL, gas CL, phi_s GDL,
phi_s CL, phi_L.
"""
struct Layout
    stage::Int
    N_CL::Int
    N_GDL::Int
    N_TOT::Int
    has_gas::Bool
    has_pot::Bool
    n_log_dofs::Int
    blocks::Vector{UnitRange{Int}}
end

function Layout(stage::Int, Ncl::Int, Ngdl::Int)
    has_gas = stage >= 3
    has_pot = stage >= 5
    n = 5 * Ncl
    blocks = UnitRange{Int}[1:n]
    if has_gas
        push!(blocks, (n+1):(n + 4*Ngdl))
        push!(blocks, (n + 4*Ngdl + 1):(n + 4*(Ngdl + Ncl)))
        n += 4 * (Ngdl + Ncl)
    end
    if has_pot
        push!(blocks, (n+1):(n + Ngdl))
        push!(blocks, (n + Ngdl + 1):(n + Ngdl + Ncl))
        push!(blocks, (n + Ngdl + Ncl + 1):(n + Ngdl + 2*Ncl))
        n += Ngdl + 2*Ncl
    end
    return Layout(stage, Ncl, Ngdl, n, has_gas, has_pot, 5*Ncl, blocks)
end

# ── Index functions (§11 Stage 5) ────────────────────────────────────────────
@inline liquid_idx(ix::Int, i::Int) = 5*(ix - 1) + i
@inline _gas_start(ly::Layout) = 5 * ly.N_CL
@inline gas_gdl_idx(ly::Layout, ig::Int, j::Int) = _gas_start(ly) + 4*(ig - 1) + j
@inline gas_cl_idx(ly::Layout, ix::Int, j::Int)  = _gas_start(ly) + 4*ly.N_GDL + 4*(ix - 1) + j
@inline _phi_s_gdl_start(ly::Layout) = _gas_start(ly) + 4*(ly.N_GDL + ly.N_CL)
@inline phi_s_gdl_idx(ly::Layout, ig::Int) = _phi_s_gdl_start(ly) + ig
@inline phi_s_cl_idx(ly::Layout, ix::Int)  = _phi_s_gdl_start(ly) + ly.N_GDL + ix
@inline phi_L_idx(ly::Layout, ix::Int)     = _phi_s_gdl_start(ly) + ly.N_GDL + ly.N_CL + ix

# ── Unpacking helpers — all return the element type of u (§9.3.2) ────────────
function unpack_log_conc(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    c = zeros(ElT, 5, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

@inline function gdl_omega(u::AbstractVector{ElT}, ly::Layout, ig::Int) where {ElT<:Real}
    b = _gas_start(ly) + 4*(ig - 1)
    @inbounds w1 = u[b+1]; @inbounds w2 = u[b+2]; @inbounds w3 = u[b+3]
    return (w1, w2, w3, omega_H2O_closure(w1, w2, w3))
end
@inline gdl_p(u::AbstractVector, ly::Layout, ig::Int) = @inbounds u[_gas_start(ly) + 4*(ig-1) + 4]

@inline function cl_omega(u::AbstractVector{ElT}, ly::Layout, ix::Int) where {ElT<:Real}
    b = _gas_start(ly) + 4*ly.N_GDL + 4*(ix - 1)
    @inbounds w1 = u[b+1]; @inbounds w2 = u[b+2]; @inbounds w3 = u[b+3]
    return (w1, w2, w3, omega_H2O_closure(w1, w2, w3))
end
@inline cl_p(u::AbstractVector, ly::Layout, ix::Int) = @inbounds u[_gas_start(ly) + 4*ly.N_GDL + 4*(ix-1) + 4]

"Gas state at combined gas-grid cell k (1..N_GDL = GDL, then the CL cells)."
@inline function gas_cell_state(u::AbstractVector{ElT}, ly::Layout, k::Int) where {ElT<:Real}
    return k <= ly.N_GDL ? (gdl_omega(u, ly, k), gdl_p(u, ly, k)) :
                           (cl_omega(u, ly, k - ly.N_GDL), cl_p(u, ly, k - ly.N_GDL))
end

function unpack_phi_s_gdl(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_GDL)
    @inbounds for ig in 1:ly.N_GDL; v[ig] = u[phi_s_gdl_idx(ly, ig)]; end
    return v
end
function unpack_phi_s_cl(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_s_cl_idx(ly, ix)]; end
    return v
end
function unpack_phi_L_cl(u::AbstractVector{ElT}, ly::Layout) where {ElT<:Real}
    v = zeros(ElT, ly.N_CL)
    @inbounds for ix in 1:ly.N_CL; v[ix] = u[phi_L_idx(ly, ix)]; end
    return v
end

"Local CL gas-phase CO2 partial pressure [atm] from the gas DOFs (§7.3)."
@inline function pCO2_from_gas(u::AbstractVector{ElT}, ly::Layout, ix::Int) where {ElT<:Real}
    w = cl_omega(u, ly, ix); p = cl_p(u, ly, ix)
    x = omega_to_x(w)
    return x[1] * p / 101325.0
end

# ── Model context ────────────────────────────────────────────────────────────
struct ModelCtx
    ly::Layout
    cl::FVMesh
    gdl::FVMesh
    gg::GasGrid
    c_eq::Vector{Float64}
end

function make_ctx(stage::Int; Ncl::Int = N_CL, Ngdl::Int = N_GDL)
    cl  = make_cl_mesh(Ncl)
    gdl = make_gdl_mesh(Ngdl)
    gg  = make_gas_grid(gdl, cl)
    ly  = Layout(stage, Ncl, Ngdl)
    return ModelCtx(ly, cl, gdl, gg, equilibrium_vector())
end

# ── The unified residual ─────────────────────────────────────────────────────
"""
    full_residual!(F, u, V_cathode, ctx)

Assembles the complete residual for the stage recorded in `ctx.ly.stage`.

`V_cathode` is **V vs RHE@pH 8** (§15.3); the conversion to SHE happens here,
exactly once.

Stage map:
  1  liquid diffusion only, Dirichlet c_eq at x = L_CL, phi_s = V_applied_SHE, phi_L = 0
  2  as 1 but Sherwood-Reynolds Neumann at x = L_CL
  3  + Stefan-Maxwell gas (u_G = 0, pressure row pinned to p_total) + R_PT
  4  + Darcy pressure equation and the convective species flux
  5  + phi_L (NP migration, monolithic DOF) and phi_s (solid Ohm's law)
"""
function full_residual!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                        V_cathode::Real, ctx::ModelCtx) where {ElT<:Real}
    ly = ctx.ly; cl = ctx.cl; gdl = ctx.gdl; gg = ctx.gg; c_eq = ctx.c_eq
    Nc = ly.N_CL; Ng = ly.N_GDL
    stage   = ly.stage
    has_gas = ly.has_gas
    has_pot = ly.has_pot
    darcy   = stage >= 4
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    c = unpack_log_conc(u, ly)

    phi_s_gdl = has_pot ? unpack_phi_s_gdl(u, ly) : zeros(ElT, 0)
    phi_s_cl  = has_pot ? unpack_phi_s_cl(u, ly)  : zeros(ElT, 0)
    phi_L     = has_pot ? unpack_phi_L_cl(u, ly)  : zeros(ElT, 0)

    # ── CO2(aq) concentration at the GDL/CL face (§7.1 Stage 1-2, §7.3 Stage 3+)
    cCO2_bc = if has_gas
        sechenov_H_CO2(view(c, :, 1)) * pCO2_from_gas(u, ly, 1)
    else
        ElT(c_CO2_henry_stage12)
    end

    # ── Step 1: liquid face fluxes, +x positive ──────────────────────────────
    J = zeros(ElT, 5, Nc + 1)
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    @inbounds J[5, 1] = has_pot ? zero(ElT) :
                        diffusion_flux(cCO2_bc, c[5, 1], D_eff[5], cl.dx_between[1])
    # J[1:4,1] stay zero — ions have no flux at the GDL/CL face (§7.1)
    @inbounds for f in 2:Nc, i in 1:5
        J[i, f] = has_pot ?
            np_flux(c[i, f-1], c[i, f], phi_L[f-1], phi_L[f],
                    D_eff[i], z_chg[i], cl.dx_between[f]) :
            diffusion_flux(c[i, f-1], c[i, f], D_eff[i], cl.dx_between[f])
    end
    @inbounds if stage == 1
        for i in 1:5                                   # Dirichlet reservoir
            J[i, Nc+1] = diffusion_flux(c[i, Nc], c_eq[i], D_eff[i], cl.dx_between[Nc+1])
        end
    elseif has_pot
        for i in 1:5                                   # §11 gotcha #12: NP at the boundary
            L_eff = D_eff[i] / k_MT[i]
            J[i, Nc+1] = np_flux(c[i, Nc], c_eq[i], phi_L[Nc], zero(ElT),
                                 D_eff[i], z_chg[i], L_eff)
        end
    else
        for i in 1:5                                   # Sherwood-Reynolds Neumann (§7.2)
            J[i, Nc+1] = -k_MT[i] * (c_eq[i] - c[i, Nc])
        end
    end

    # ── Step 2: faradaic rates and phase transfer per CL cell ────────────────
    R_co2r = zeros(ElT, Nc); R_her = zeros(ElT, Nc); R_pt = zeros(ElT, Nc)
    lam = lambda_MHC(); i0 = i0_CO2R(); kgl = k_GL()
    @inbounds for ix in 1:Nc
        phis = has_pot ? phi_s_cl[ix] : ElT(V_applied_SHE)
        phil = has_pot ? phi_L[ix]    : zero(ElT)
        R_co2r[ix] = R_CO2R(c[5, ix], phis, phil, c[1, ix], a_v_CL, lam, i0, n_e)
        R_her[ix]  = R_HER(phis, phil, c[1, ix], a_v_CL, n_e)
        if has_gas
            H_local = sechenov_H_CO2(view(c, :, ix))
            R_pt[ix] = a_v_GL * kgl * (H_local * pCO2_from_gas(u, ly, ix) - c[5, ix])
        end
    end

    # ── Step 3: liquid species residuals, integrated FV form (§3.2, §6.4) ────
    @inbounds for ix in 1:Nc
        buf = buffer_sources(view(c, :, ix))
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2
                src += 2*R_co2r[ix] + 2*R_her[ix]          # OH- from both reactions
            elseif i == 5
                src += R_pt[ix] - R_co2r[ix]               # CO2(aq): +R_PT, -R_CO2R
            end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * cl.dx[ix]
        end
    end

    # ── Step 4: gas-phase residuals (§6) ─────────────────────────────────────
    if has_gas
        JG = zeros(ElT, 3, gg.N + 1)
        RU = zeros(ElT, gg.N + 1)
        # face 1: flow-field Dirichlet ghost (x_inlet as mass fractions, p_total)
        wR1, pR1 = gas_cell_state(u, ly, 1)
        f1 = gas_face_flux(inlet_omega(ElT), ElT(p_total), wR1, pR1,
                           gg.Dface[1], gg.kappa[1], gg.dxc[1], darcy)
        @inbounds JG[1,1] = f1[1]; @inbounds JG[2,1] = f1[2]
        @inbounds JG[3,1] = f1[3]; @inbounds RU[1]   = f1[4]
        @inbounds for f in 2:gg.N
            wL, pL = gas_cell_state(u, ly, f-1)
            wR, pR = gas_cell_state(u, ly, f)
            ff = gas_face_flux(wL, pL, wR, pR, gg.Dface[f], gg.kappa[f], gg.dxc[f], darcy)
            JG[1,f] = ff[1]; JG[2,f] = ff[2]; JG[3,f] = ff[3]; RU[f] = ff[4]
        end
        # face gg.N+1 (x = L_CL): zero species flux and no gas flow — left at zero.

        @inbounds for ig in 1:Ng                       # GDL: no sources
            for j in 1:3
                F[gas_gdl_idx(ly, ig, j)] = JG[j, ig] - JG[j, ig+1]
            end
            F[gas_gdl_idx(ly, ig, 4)] = darcy ? (RU[ig] - RU[ig+1]) :
                                                (gdl_p(u, ly, ig) - p_total)
        end
        @inbounds for ix in 1:Nc                       # CL: §6.5 sources
            fL = Ng + ix; fR = Ng + ix + 1
            S1 = -R_pt[ix]   * M_gas[1]
            S2 =  R_co2r[ix] * M_gas[2]
            S3 =  R_her[ix]  * M_gas[3]
            F[gas_cl_idx(ly, ix, 1)] = JG[1,fL] - JG[1,fR] + S1 * cl.dx[ix]
            F[gas_cl_idx(ly, ix, 2)] = JG[2,fL] - JG[2,fR] + S2 * cl.dx[ix]
            F[gas_cl_idx(ly, ix, 3)] = JG[3,fL] - JG[3,fR] + S3 * cl.dx[ix]
            F[gas_cl_idx(ly, ix, 4)] = darcy ?
                (RU[fL] - RU[fR] + (S1 + S2 + S3) * cl.dx[ix]) :
                (cl_p(u, ly, ix) - p_total)
        end
    end

    # ── Steps 5 and 6: potentials ────────────────────────────────────────────
    if has_pot
        # ---- phi_L: local ionic charge balance per CL cell (§3.1) -----------
        i_L = zeros(ElT, Nc + 1)
        @inbounds for f in 1:(Nc+1)
            s = zero(ElT)
            for i in 1:5
                s += z_chg[i] * J[i, f]
            end
            i_L[f] = F_const * s
        end
        # Cs+ migration current — Cs+ has no residual row of its own but its NP
        # current MUST enter i_L (§3.1, §9.4 item 8).  No flux at either CL face:
        # only the Nc-1 interior faces get a np_flux contribution.
        @inbounds for f in 2:Nc
            cCsL = cCs_from_electroneutrality(view(c, :, f-1))
            cCsR = cCs_from_electroneutrality(view(c, :, f))
            J_Cs = np_flux(cCsL, cCsR, phi_L[f-1], phi_L[f],
                           D_eff_Cs, z_Cs, cl.dx_between[f])
            i_L[f] += F_const * z_Cs * J_Cs
        end
        @inbounds for ix in 1:Nc
            i_far_vol = n_e * F_const * (R_co2r[ix] + R_her[ix])
            F[phi_L_idx(ly, ix)] = (i_L[ix+1] - i_L[ix]) + i_far_vol * cl.dx[ix]
        end
        # Dirichlet gauge override at the electrolyte-side cell (§3.1, gotcha #7)
        @inbounds F[phi_L_idx(ly, Nc)] = phi_L[Nc]

        # ---- phi_s: solid Ohm's law on GDL + CL (§3.4) ----------------------
        i_s = zeros(ElT, Ng + Nc + 1)
        # Ghost-cell Dirichlet at the current collector — NOT a cell-1 override.
        @inbounds i_s[1] = -sigma_s_eff_GDL * (phi_s_gdl[1] - V_applied_SHE) / gdl.dx_between[1]
        @inbounds for f in 2:Ng
            i_s[f] = -sigma_s_eff_GDL * (phi_s_gdl[f] - phi_s_gdl[f-1]) / gdl.dx_between[f]
        end
        sig_int = 2.0 * sigma_s_eff_GDL * sigma_s_eff_CL / (sigma_s_eff_GDL + sigma_s_eff_CL)
        d_int   = gdl.dx_between[Ng+1] + cl.dx_between[1]
        @inbounds i_s[Ng+1] = -sig_int * (phi_s_cl[1] - phi_s_gdl[Ng]) / d_int
        @inbounds for f in 2:Nc
            i_s[Ng+f] = -sigma_s_eff_CL * (phi_s_cl[f] - phi_s_cl[f-1]) / cl.dx_between[f]
        end
        # i_s[Ng+Nc+1] = 0 — Neumann at the electrolyte side, left at zero.

        @inbounds for ig in 1:Ng
            F[phi_s_gdl_idx(ly, ig)] = i_s[ig+1] - i_s[ig]
        end
        @inbounds for ix in 1:Nc
            i_far_vol = n_e * F_const * (R_co2r[ix] + R_her[ix])
            F[phi_s_cl_idx(ly, ix)] = (i_s[Ng+1+ix] - i_s[Ng+ix]) - i_far_vol * cl.dx[ix]
        end
    end

    return F
end

# ── §17.4 Current extractors ─────────────────────────────────────────────────
"""
    compute_rates(u, V_cathode, ctx) -> (R_CO2R, R_HER) per CL cell [mol/m^3/s]

Stages 1-4 use `phi_s = V_applied_SHE`, `phi_L = 0`; Stage 5 reads the phi_s
and phi_L DOFs.
"""
function compute_rates(u::Vector{Float64}, V_cathode::Float64, ctx::ModelCtx)
    ly = ctx.ly; Nc = ly.N_CL
    c = unpack_log_conc(u, ly)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    phi_s_cl = ly.has_pot ? unpack_phi_s_cl(u, ly) : Float64[]
    phi_L    = ly.has_pot ? unpack_phi_L_cl(u, ly) : Float64[]
    rc = zeros(Nc); rh = zeros(Nc)
    for ix in 1:Nc
        phis = ly.has_pot ? phi_s_cl[ix] : V_applied_SHE
        phil = ly.has_pot ? phi_L[ix]    : 0.0
        rc[ix] = R_CO2R(c[5,ix], phis, phil, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        rh[ix] = R_HER(phis, phil, c[1,ix], a_v_CL, n_e)
    end
    return rc, rh
end

"Partial geometric current densities [A/m^2]: i = n_e F integral(R) dx (§17.4)."
function compute_currents(u::Vector{Float64}, V_cathode::Float64, ctx::ModelCtx)
    rc, rh = compute_rates(u, V_cathode, ctx)
    i_CO = 0.0; i_H2 = 0.0
    for ix in 1:ctx.ly.N_CL
        i_CO += n_e * F_const * rc[ix] * ctx.cl.dx[ix]
        i_H2 += n_e * F_const * rh[ix] * ctx.cl.dx[ix]
    end
    return (i_CO, i_H2, i_CO + i_H2)
end

compute_i_CO(u, V_cathode, ctx) = compute_currents(u, V_cathode, ctx)[1]
compute_i_H2(u, V_cathode, ctx) = compute_currents(u, V_cathode, ctx)[2]
