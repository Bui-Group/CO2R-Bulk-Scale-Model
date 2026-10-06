# =============================================================================
# assembly.jl — residual assembly and current extraction (BaseCase.md §3, §11, §17.4)
# Stage 1: liquid Nernst–Planck DIFFUSION ONLY + Dirichlet both sides + simplified
#          kinetics (phi_s = V_applied_SHE, phi_L = 0).  5×N_CL DOFs, cell-major.
# =============================================================================

# Liquid DOF index: species i (1=H,2=OH,3=HCO3,4=CO3,5=CO2) at CL cell ix.
@inline liquid_idx(ix::Int, i::Int) = 5*(ix - 1) + i

# §9.5.3 weight blocks for each stage's DOF layout:
#   liquid log-c | gas GDL | gas CL | φ_s GDL | φ_s CL | φ_L   (later stages append blocks)
function dof_blocks(stage::Int)
    blocks = UnitRange{Int}[1:5*N_CL]
    if stage >= 3
        push!(blocks, gas_gdl_idx(1,1):gas_gdl_idx(N_GDL,4))
        push!(blocks, gas_cl_idx(1,1):gas_cl_idx(N_CL,4))
    end
    if stage >= 5
        push!(blocks, phi_s_gdl_idx(1):phi_s_gdl_idx(N_GDL))
        push!(blocks, phi_s_cl_idx(1):phi_s_cl_idx(N_CL))
        push!(blocks, phi_L_idx(1):phi_L_idx(N_CL))
    end
    return blocks
end

# Unpack log-concentration DOFs -> physical concentration matrix (5×N), type-generic.
function unpack_log_conc(u, N::Int)
    ElT = eltype(u)
    c = Matrix{ElT}(undef, 5, N)
    @inbounds for ix in 1:N, i in 1:5
        c[i, ix] = exp(clamp(u[liquid_idx(ix, i)], -50.0, 50.0))
    end
    return c
end

"""
    residual_stage1!(F, u, V_cathode, cl_mesh, c_eq, c_CO2_bc)

Stage 1 residual (§11 Stage 1). Integrated FV form  F = J_left − J_right + S·dx.
Pure diffusion; CO₂(aq) Dirichlet via Henry at the GDL face, ions no-flux there;
all species Dirichlet to bulk equilibrium `c_eq` at the electrolyte face.
"""
function residual_stage1!(F, u, V_cathode::Real, cl_mesh, c_eq, c_CO2_bc::Real)
    N = cl_mesh.N
    ElT = eltype(u)
    dx  = cl_mesh.dx
    dxb = cl_mesh.dx_between                      # length N+1
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    c = unpack_log_conc(u, N)                     # 5×N

    # ── diffusion fluxes at the N+1 faces
    J = zeros(ElT, 5, N+1)
    # left boundary face (f=1): CO₂(aq) Dirichlet (Henry); ions no-flux (stay 0)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], dxb[1])
    # interior faces f=2..N (between cell f-1 and f)
    @inbounds for f in 2:N, i in 1:5
        J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f])
    end
    # right boundary face (f=N+1): Dirichlet to bulk equilibrium c_eq
    @inbounds for i in 1:5
        J[i, N+1] = diffusion_flux(c[i, N], c_eq[i], D_eff[i], dxb[N+1])
    end

    # ── cell residuals (integrated FV)
    @inbounds for ix in 1:N
        cH = c[1, ix]; cCO2 = c[5, ix]
        r_co2r = R_CO2R(cCO2, V_applied_SHE, zero(ElT), cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), cH, a_v_CL, n_e)
        (S_H, S_OH, S_HCO3, S_CO3, S_CO2) = buffer_sources(c[1,ix], c[2,ix], c[3,ix], c[4,ix], c[5,ix])
        buf = (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end   # OH⁻ produced by CO₂R and HER
            if i == 5; src -= r_co2r;             end   # CO₂(aq) consumed by CO₂R
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end
    return F
end

"""
    residual_stage2!(F, u, V_cathode, cl_mesh, c_eq, c_CO2_bc)

Stage 2 residual (§11 Stage 2). IDENTICAL to `residual_stage1!` except the
electrolyte-side (x=L_CL) face flux: the Dirichlet is replaced by the
Sherwood-Reynolds convective Neumann  J[i,N+1] = −k_MT[i]·(c_eq[i] − c[i,N])
(§7.2). `k_MT` is built from bare-water D_w in params.jl (NOT D_eff).
"""
function residual_stage2!(F, u, V_cathode::Real, cl_mesh, c_eq, c_CO2_bc::Real)
    N = cl_mesh.N
    ElT = eltype(u)
    dx  = cl_mesh.dx
    dxb = cl_mesh.dx_between
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    c = unpack_log_conc(u, N)

    J = zeros(ElT, 5, N+1)
    J[5, 1] = diffusion_flux(c_CO2_bc, c[5, 1], D_eff[5], dxb[1])     # GDL-side CO₂ Dirichlet; ions no-flux
    @inbounds for f in 2:N, i in 1:5
        J[i, f] = diffusion_flux(c[i, f-1], c[i, f], D_eff[i], dxb[f])
    end
    # ── electrolyte-side: Sherwood-Reynolds convective Neumann (the ONLY change vs Stage 1)
    @inbounds for i in 1:5
        J[i, N+1] = -k_MT[i] * (c_eq[i] - c[i, N])
    end

    @inbounds for ix in 1:N
        cH = c[1, ix]; cCO2 = c[5, ix]
        r_co2r = R_CO2R(cCO2, V_applied_SHE, zero(ElT), cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        r_her  = R_HER(V_applied_SHE, zero(ElT), cH, a_v_CL, n_e)
        (S_H, S_OH, S_HCO3, S_CO3, S_CO2) = buffer_sources(c[1,ix], c[2,ix], c[3,ix], c[4,ix], c[5,ix])
        buf = (S_H, S_OH, S_HCO3, S_CO3, S_CO2)
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*r_co2r + 2*r_her; end
            if i == 5; src -= r_co2r;             end
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * dx[ix]
        end
    end
    return F
end

# ---------------------------------------------------------------------------
# Current extractors (§17.4). Partial geometric current densities [A/m²].
# Stages 1–4: pass phi_s = V_applied_SHE, phi_L = 0.
# ---------------------------------------------------------------------------
function compute_i_CO(u::Vector{Float64}, V_applied_SHE::Float64, cl_mesh)
    c = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    @inbounds for ix in 1:cl_mesh.N
        r = R_CO2R(c[5,ix], V_applied_SHE, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i      # A/m²
end

function compute_i_H2(u::Vector{Float64}, V_applied_SHE::Float64, cl_mesh)
    c = unpack_log_conc(u, cl_mesh.N)
    i = 0.0
    @inbounds for ix in 1:cl_mesh.N
        r = R_HER(V_applied_SHE, 0.0, c[1,ix], a_v_CL, n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i      # A/m²
end

compute_i_total(u, V_applied_SHE, cl_mesh) =
    compute_i_CO(u, V_applied_SHE, cl_mesh) + compute_i_H2(u, V_applied_SHE, cl_mesh)

# =============================================================================
# Stage 3: Stefan-Maxwell gas transport + R_PT phase transfer (§6, §7.3, §7.4, §11)
#   Adds 4×(N_GDL+N_CL) gas DOFs. u_G = 0 (no Darcy convection); pressure pinned to
#   p_total. Liquid CO₂(aq) GDL-face Dirichlet now = Henry(local CL gas cell 1).
#   Simplified kinetics still (phi_s = V_applied_SHE, phi_L = 0).
# =============================================================================

# Gas-phase residual block (combined GDL+CL gas mesh), filled into F in place.
function residual_gas_stage3!(F, u, gdl_mesh, cl_mesh, Rco, Rh2, Rpt, uG::Real)
    ElT = eltype(u); NG = gdl_mesh.N; N = cl_mesh.N; NC = NG + N
    getcell(gc) = gc <= NG ? (gdl_omega(u,gc), gdl_p(u,gc)) : (cl_omega(u,gc-NG), cl_p(u,gc-NG))
    celldx(gc)  = gc <= NG ? gdl_mesh.dx[gc] : cl_mesh.dx[gc-NG]

    function faceJ(f)                       # SM mass flux (3-tuple) at combined face f (1..NC)
        if f == 1                            # inlet (x=−L_GDL): Dirichlet inlet composition + p_total
            omL = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4]); pL = p_total
            omR, pR = getcell(1); Dbin = D_bin_GDL; dxcc = gdl_mesh.dx_between[1]
        elseif f <= NG                       # interior GDL faces
            omL, pL = getcell(f-1); omR, pR = getcell(f); Dbin = D_bin_GDL; dxcc = gdl_mesh.dx_between[f]
        elseif f == NG+1                     # GDL/CL interface
            omL, pL = getcell(NG); omR, pR = getcell(NG+1); Dbin = D_bin_IF
            dxcc = cl_mesh.x_centers[1] - gdl_mesh.x_centers[NG]
        else                                  # interior CL faces  (f in NG+2..NC)
            omL, pL = getcell(f-1); omR, pR = getcell(f); Dbin = D_bin_CL; dxcc = cl_mesh.dx_between[f-NG]
        end
        return sm_face_flux(omL, omR, pL, pR, Dbin, dxcc, uG)
    end

    Jprev = faceJ(1)
    for gc in 1:NC
        Jnext = gc < NC ? faceJ(gc+1) : (zero(ElT), zero(ElT), zero(ElT))   # electrolyte face: zero flux
        dxg = celldx(gc)
        if gc <= NG
            S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
            i1 = gas_gdl_idx(gc,1); i2 = gas_gdl_idx(gc,2); i3 = gas_gdl_idx(gc,3); ip = gas_gdl_idx(gc,4)
            pcell = gdl_p(u, gc)
        else
            ix = gc - NG
            S1 = -Rpt[ix]*M_gas[1]; S2 = Rco[ix]*M_gas[2]; S3 = Rh2[ix]*M_gas[3]
            i1 = gas_cl_idx(ix,1); i2 = gas_cl_idx(ix,2); i3 = gas_cl_idx(ix,3); ip = gas_cl_idx(ix,4)
            pcell = cl_p(u, ix)
        end
        F[i1] = Jprev[1] - Jnext[1] + S1*dxg
        F[i2] = Jprev[2] - Jnext[2] + S2*dxg
        F[i3] = Jprev[3] - Jnext[3] + S3*dxg
        F[ip] = pcell - p_total                       # Stage 3: pressure pinned (Darcy off)
        Jprev = Jnext
    end
    return F
end

"""
    residual_stage3!(F, u, V_cathode, cl_mesh, gdl_mesh, c_eq)

Stage 3 monolithic residual: Stage-2 liquid (Sherwood right BC) + gas DOFs (SM, u_G=0,
p pinned) + volumetric CO₂ phase transfer R_PT. CO₂(aq) GDL-face Dirichlet = Henry(local gas).
"""
function residual_stage3!(F, u, V_cathode::Real, cl_mesh, gdl_mesh, c_eq)
    ElT = eltype(u); N = cl_mesh.N
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)

    # per-CL-cell faradaic rates + phase transfer
    Rco = zeros(ElT, N); Rh2 = zeros(ElT, N); Rpt = zeros(ElT, N)
    @inbounds for ix in 1:N
        cH = c[1,ix]; cCO2 = c[5,ix]
        Rco[ix] = R_CO2R(cCO2, V_applied_SHE, zero(ElT), cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rh2[ix] = R_HER(V_applied_SHE, zero(ElT), cH, a_v_CL, n_e)
        Hloc = H_eff_sechenov(c[1,ix], c[2,ix], c[3,ix], c[4,ix])
        Rpt[ix] = a_v_GL * k_GL() * (Hloc * pCO2_cl(u, ix) - cCO2)      # §7.4, +R_PT into liquid
    end

    # ── liquid residual (Sherwood right BC; CO₂ left BC = Henry from CL gas cell 1)
    dx = cl_mesh.dx; dxb = cl_mesh.dx_between
    cCO2_ghost = H_eff_sechenov(c[1,1], c[2,1], c[3,1], c[4,1]) * pCO2_cl(u, 1)
    J = zeros(ElT, 5, N+1)
    J[5,1] = diffusion_flux(cCO2_ghost, c[5,1], D_eff[5], dxb[1])       # ions no-flux at GDL (stay 0)
    @inbounds for f in 2:N, i in 1:5
        J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], dxb[f])
    end
    @inbounds for i in 1:5
        J[i,N+1] = -k_MT[i] * (c_eq[i] - c[i,N])
    end
    @inbounds for ix in 1:N
        buf = buffer_sources(c[1,ix], c[2,ix], c[3,ix], c[4,ix], c[5,ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*Rco[ix] + 2*Rh2[ix]; end
            if i == 5; src -= Rco[ix]; src += Rpt[ix]; end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * dx[ix]
        end
    end

    # ── gas residual (u_G = 0 in Stage 3)
    residual_gas_stage3!(F, u, gdl_mesh, cl_mesh, Rco, Rh2, Rpt, zero(ElT))
    return F
end

# Structural sparsity pattern for Stage 3/4 (block layout liquid|GDLgas|CLgas).
function build_gas_pattern(N_CL_::Int, N_GDL_::Int)
    rows = Int[]; cols = Int[]
    add!(r, cc) = (push!(rows, r); push!(cols, cc))
    # liquid–liquid block-tridiagonal (5×5)
    for ix in 1:N_CL_, jx in max(1,ix-1):min(N_CL_,ix+1), a in 1:5, b in 1:5
        add!(liquid_idx(ix,a), liquid_idx(jx,b))
    end
    # liquid[ix] <-> CL-gas[ix] local coupling (R_PT, faradaic, CO₂ left BC)
    for ix in 1:N_CL_, a in 1:5, b in 1:4
        add!(liquid_idx(ix,a), gas_cl_idx(ix,b))
        add!(gas_cl_idx(ix,b), liquid_idx(ix,a))
    end
    # gas–gas block-tridiagonal (4×4) over the combined GDL+CL gas mesh
    gci(gc, j) = gc <= N_GDL_ ? gas_gdl_idx(gc,j) : gas_cl_idx(gc-N_GDL_,j)
    NC = N_GDL_ + N_CL_
    for gc in 1:NC, gd in max(1,gc-1):min(NC,gc+1), a in 1:4, b in 1:4
        add!(gci(gc,a), gci(gd,b))
    end
    return rows, cols
end

# Stage 3 warm-start: Stage-2 liquid + inlet gas (ω_inlet, p_total) everywhere.
function make_u0_stage3(u2::Vector{Float64})
    u0 = zeros(NDOF_GAS)
    u0[1:5*N_CL] .= u2
    for ig in 1:N_GDL
        u0[gas_gdl_idx(ig,1)] = omega_inlet[1]; u0[gas_gdl_idx(ig,2)] = omega_inlet[2]
        u0[gas_gdl_idx(ig,3)] = omega_inlet[3]; u0[gas_gdl_idx(ig,4)] = p_total
    end
    for ix in 1:N_CL
        u0[gas_cl_idx(ix,1)] = omega_inlet[1]; u0[gas_cl_idx(ix,2)] = omega_inlet[2]
        u0[gas_cl_idx(ix,3)] = omega_inlet[3]; u0[gas_cl_idx(ix,4)] = p_total
    end
    return u0
end

# =============================================================================
# Stage 4: finalize Darcy pressure (§6.2, §7.4, §11 Stage 4)
#   Same 1200 DOFs as Stage 3. The pressure DOF residual becomes the true Darcy
#   continuity balance (ρu_L − ρu_R + S_mass·dx); species flux gains the convective
#   term ρ ω_j u_G with u_G = −(κ/μ)·dp/dx. Liquid block IDENTICAL to Stage 3.
# =============================================================================

# Gas residual with Darcy convection + continuity. κ per face: GDL / harmonic-mean IF / CL.
function residual_gas_stage4!(F, u, gdl_mesh, cl_mesh, Rco, Rh2, Rpt)
    ElT = eltype(u); NG = gdl_mesh.N; N = cl_mesh.N; NC = NG + N
    getcell(gc) = gc <= NG ? (gdl_omega(u,gc), gdl_p(u,gc)) : (cl_omega(u,gc-NG), cl_p(u,gc-NG))
    celldx(gc)  = gc <= NG ? gdl_mesh.dx[gc] : cl_mesh.dx[gc-NG]

    function gasface(f)        # returns (J_CO2, J_CO, J_H2, ρu) at combined face f
        if f == 1
            omL = (omega_inlet[1], omega_inlet[2], omega_inlet[3], omega_inlet[4]); pL = p_total
            omR, pR = getcell(1); Dbin = D_bin_GDL; kap = kappa_GDL; dxcc = gdl_mesh.dx_between[1]
        elseif f <= NG
            omL, pL = getcell(f-1); omR, pR = getcell(f); Dbin = D_bin_GDL; kap = kappa_GDL; dxcc = gdl_mesh.dx_between[f]
        elseif f == NG+1
            omL, pL = getcell(NG); omR, pR = getcell(NG+1); Dbin = D_bin_IF; kap = kappa_IF
            dxcc = cl_mesh.x_centers[1] - gdl_mesh.x_centers[NG]
        else
            omL, pL = getcell(f-1); omR, pR = getcell(f); Dbin = D_bin_CL; kap = kappa_CL; dxcc = cl_mesh.dx_between[f-NG]
        end
        uG = -(kap / mu_gas) * (pR - pL) / dxcc                          # Darcy velocity
        J  = sm_face_flux(omL, omR, pL, pR, Dbin, dxcc, uG)              # SM flux incl. convection
        omf = (0.5*(omL[1]+omR[1]), 0.5*(omL[2]+omR[2]), 0.5*(omL[3]+omR[3]), 0.5*(omL[4]+omR[4]))
        rhof = 0.5*(pL+pR) * Mn_from_omega(omf) / (R_gas * T)
        return (J[1], J[2], J[3], rhof * uG)
    end

    Fp = gasface(1)
    for gc in 1:NC
        Fn = gc < NC ? gasface(gc+1) : (zero(ElT), zero(ElT), zero(ElT), zero(ElT))   # outlet: dp/dx=0 ⇒ ρu=0, J=0
        dxg = celldx(gc)
        if gc <= NG
            S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT); Smass = zero(ElT)
            i1 = gas_gdl_idx(gc,1); i2 = gas_gdl_idx(gc,2); i3 = gas_gdl_idx(gc,3); ip = gas_gdl_idx(gc,4)
        else
            ix = gc - NG
            S1 = -Rpt[ix]*M_gas[1]; S2 = Rco[ix]*M_gas[2]; S3 = Rh2[ix]*M_gas[3]; Smass = S1 + S2 + S3
            i1 = gas_cl_idx(ix,1); i2 = gas_cl_idx(ix,2); i3 = gas_cl_idx(ix,3); ip = gas_cl_idx(ix,4)
        end
        F[i1] = Fp[1] - Fn[1] + S1*dxg
        F[i2] = Fp[2] - Fn[2] + S2*dxg
        F[i3] = Fp[3] - Fn[3] + S3*dxg
        F[ip] = Fp[4] - Fn[4] + Smass*dxg          # Darcy continuity (§6.4): ρu_L − ρu_R + S_mass·dx
        Fp = Fn
    end
    return F
end

"""
    residual_stage4!(F, u, V_cathode, cl_mesh, gdl_mesh, c_eq)

Stage 4 monolithic residual: Stage-3 liquid + R_PT (unchanged) with the gas block
upgraded to the full Darcy momentum/continuity formulation.
"""
function residual_stage4!(F, u, V_cathode::Real, cl_mesh, gdl_mesh, c_eq)
    ElT = eltype(u); N = cl_mesh.N
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)

    # per-CL-cell faradaic rates + phase transfer (identical to Stage 3)
    Rco = zeros(ElT, N); Rh2 = zeros(ElT, N); Rpt = zeros(ElT, N)
    @inbounds for ix in 1:N
        cH = c[1,ix]; cCO2 = c[5,ix]
        Rco[ix] = R_CO2R(cCO2, V_applied_SHE, zero(ElT), cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rh2[ix] = R_HER(V_applied_SHE, zero(ElT), cH, a_v_CL, n_e)
        Hloc = H_eff_sechenov(c[1,ix], c[2,ix], c[3,ix], c[4,ix])
        Rpt[ix] = a_v_GL * k_GL() * (Hloc * pCO2_cl(u, ix) - cCO2)
    end

    # ── liquid residual (identical to Stage 3: Sherwood right BC, Henry-from-gas left BC, +R_PT)
    dx = cl_mesh.dx; dxb = cl_mesh.dx_between
    cCO2_ghost = H_eff_sechenov(c[1,1], c[2,1], c[3,1], c[4,1]) * pCO2_cl(u, 1)
    J = zeros(ElT, 5, N+1)
    J[5,1] = diffusion_flux(cCO2_ghost, c[5,1], D_eff[5], dxb[1])
    @inbounds for f in 2:N, i in 1:5
        J[i,f] = diffusion_flux(c[i,f-1], c[i,f], D_eff[i], dxb[f])
    end
    @inbounds for i in 1:5
        J[i,N+1] = -k_MT[i] * (c_eq[i] - c[i,N])
    end
    @inbounds for ix in 1:N
        buf = buffer_sources(c[1,ix], c[2,ix], c[3,ix], c[4,ix], c[5,ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*Rco[ix] + 2*Rh2[ix]; end
            if i == 5; src -= Rco[ix]; src += Rpt[ix]; end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * dx[ix]
        end
    end

    # ── gas residual with Darcy convection + continuity
    residual_gas_stage4!(F, u, gdl_mesh, cl_mesh, Rco, Rh2, Rpt)
    return F
end

# =============================================================================
# Stage 5: potentials φ_L (NP migration) + φ_s (solid Ohm's law) (§3.1, §3.4, §11 Stage 5)
#   Adds N_GDL+N_CL φ_s DOFs and N_CL φ_L DOFs → 1480 total. Uses FITTED kinetics.
#   Kinetics driving force becomes local (φ_s_cl − φ_L) − U_eq.
# =============================================================================

# Electronic current at the N_GDL+N_CL+1 solid faces (§3.4). i_s = −σ_eff·∇φ_s.
# Collector (face 1): ghost-cell Dirichlet φ_s = V_applied_SHE (NOT a residual override).
# Electrolyte (last face): Neumann i_s = 0. Harmonic-mean σ at the GDL/CL interface.
function solid_face_currents(u, gdl_mesh, cl_mesh, V_applied_SHE::Real)
    ElT = eltype(u); NG = gdl_mesh.N; N = cl_mesh.N; NC = NG + N
    is = zeros(ElT, NC + 1)
    is[1] = -sigma_s_eff_GDL * (u[phi_s_gdl_idx(1)] - V_applied_SHE) / gdl_mesh.dx_between[1]
    @inbounds for f in 2:NG
        is[f] = -sigma_s_eff_GDL * (u[phi_s_gdl_idx(f)] - u[phi_s_gdl_idx(f-1)]) / gdl_mesh.dx_between[f]
    end
    dxcc_if = cl_mesh.x_centers[1] - gdl_mesh.x_centers[NG]
    is[NG+1] = -sigma_s_eff_IF * (u[phi_s_cl_idx(1)] - u[phi_s_gdl_idx(NG)]) / dxcc_if
    @inbounds for f in NG+2:NC
        ix = f - NG
        is[f] = -sigma_s_eff_CL * (u[phi_s_cl_idx(ix)] - u[phi_s_cl_idx(ix-1)]) / cl_mesh.dx_between[ix]
    end
    is[NC+1] = zero(ElT)                  # electrolyte: Neumann zero
    return is
end

"""
    residual_stage5!(F, u, V_cathode, cl_mesh, gdl_mesh, c_eq)

Full coupled Stage 5 residual. Liquid NP with migration (np_flux incl. Cs⁺ current in i_L),
gas Darcy (Stage 4 block), φ_L charge balance (+ gauge override), φ_s solid Ohm (+ ghost BC).
Kinetics use LOCAL (φ_s_cl, φ_L). FITTED kinetics loaded via the Ref accessors.
"""
function residual_stage5!(F, u, V_cathode::Real, cl_mesh, gdl_mesh, c_eq)
    ElT = eltype(u); N = cl_mesh.N; NG = gdl_mesh.N
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset
    c = unpack_log_conc(u, N)
    dx = cl_mesh.dx; dxb = cl_mesh.dx_between

    # faradaic rates (LOCAL φ_s_cl, φ_L) + phase transfer
    Rco = zeros(ElT, N); Rh2 = zeros(ElT, N); Rpt = zeros(ElT, N)
    @inbounds for ix in 1:N
        cH = c[1,ix]; cCO2 = c[5,ix]
        phis = u[phi_s_cl_idx(ix)]; phil = u[phi_L_idx(ix)]
        Rco[ix] = R_CO2R(cCO2, phis, phil, cH, a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rh2[ix] = R_HER(phis, phil, cH, a_v_CL, n_e)
        Hloc = H_eff_sechenov(c[1,ix], c[2,ix], c[3,ix], c[4,ix])
        Rpt[ix] = a_v_GL * k_GL() * (Hloc * pCO2_cl(u, ix) - cCO2)
    end

    # ── liquid NP fluxes (diffusion + migration); CO₂ (z=0) reduces to diffusion
    cCO2_ghost = H_eff_sechenov(c[1,1], c[2,1], c[3,1], c[4,1]) * pCO2_cl(u, 1)
    J = zeros(ElT, 5, N+1)
    J[5,1] = diffusion_flux(cCO2_ghost, c[5,1], D_eff[5], dxb[1])     # ions no-flux at GDL (J[1:4,1]=0)
    @inbounds for f in 2:N, i in 1:5
        J[i,f] = np_flux(c[i,f-1], c[i,f], u[phi_L_idx(f-1)], u[phi_L_idx(f)], D_eff[i], z_charge[i], dxb[f])
    end
    @inbounds for i in 1:5                                            # right BC: NP migration to c_eq (φ_L=0 gauge)
        L_eff = D_eff[i] / k_MT[i]
        J[i,N+1] = np_flux(c[i,N], c_eq[i], u[phi_L_idx(N)], zero(ElT), D_eff[i], z_charge[i], L_eff)
    end

    @inbounds for ix in 1:N                                          # liquid species residuals
        buf = buffer_sources(c[1,ix], c[2,ix], c[3,ix], c[4,ix], c[5,ix])
        for i in 1:5
            src = eps_I * buf[i]
            if i == 2; src += 2*Rco[ix] + 2*Rh2[ix]; end
            if i == 5; src -= Rco[ix]; src += Rpt[ix]; end
            F[liquid_idx(ix,i)] = J[i,ix] - J[i,ix+1] + src * dx[ix]
        end
    end

    # ── gas residual (Darcy, Stage 4 block; uses the φ_s/φ_L-aware Rco/Rh2/Rpt)
    residual_gas_stage4!(F, u, gdl_mesh, cl_mesh, Rco, Rh2, Rpt)

    # ── φ_L residual: per-cell ionic charge balance (§3.1). Cs⁺ migration MUST enter i_L (gotcha #9b).
    cCs_vec = [c[2,ix] + c[3,ix] + 2*c[4,ix] - c[1,ix] - C_fix_eff for ix in 1:N]
    JCs = zeros(ElT, N+1)                                            # JCs[1]=JCs[N+1]=0 (no Cs flux at either CL face)
    @inbounds for f in 2:N
        JCs[f] = np_flux(cCs_vec[f-1], cCs_vec[f], u[phi_L_idx(f-1)], u[phi_L_idx(f)], D_eff_Cs, z_Cs, dxb[f])
    end
    iL = zeros(ElT, N+1)
    @inbounds for f in 1:N+1
        s = zero(ElT)
        for i in 1:5; s += z_charge[i] * J[i,f]; end
        iL[f] = F_const * (s + z_Cs * JCs[f])
    end
    @inbounds for ix in 1:N
        i_far = n_e * F_const * (Rco[ix] + Rh2[ix])
        F[phi_L_idx(ix)] = (iL[ix+1] - iL[ix]) + i_far * dx[ix]
    end
    F[phi_L_idx(N)] = u[phi_L_idx(N)]                                # gauge override: φ_L=0 at electrolyte

    # ── φ_s residual: solid Ohm + ghost-cell Dirichlet at collector, −i_far_vol·dx in CL (§3.4)
    is = solid_face_currents(u, gdl_mesh, cl_mesh, V_applied_SHE)
    @inbounds for ig in 1:NG
        F[phi_s_gdl_idx(ig)] = is[ig+1] - is[ig]
    end
    @inbounds for ix in 1:N
        i_far = n_e * F_const * (Rco[ix] + Rh2[ix])
        F[phi_s_cl_idx(ix)] = (is[NG+1+ix] - is[NG+ix]) - i_far * dx[ix]
    end
    return F
end

# Stage 5 warm-start: fit-cache liquid+gas, φ_s=V_applied_SHE everywhere, φ_L=0 (§9.8, gotcha #9/#9a).
function make_u0_stage5(u_fit::Vector{Float64}, V_cathode::Float64)
    u0 = zeros(NDOF_STAGE5)
    u0[1:NDOF_GAS] .= u_fit
    Vapp = V_cathode - V_RHE_to_SHE_offset
    for ig in 1:N_GDL; u0[phi_s_gdl_idx(ig)] = Vapp; end
    for ix in 1:N_CL;  u0[phi_s_cl_idx(ix)]  = Vapp; end
    return u0                                                        # φ_L already 0
end

# Stage 5 anchor guess at V_c = 0 (§9.5.5): constructed from physics, never from a cache —
# bulk buffer equilibrium in every CL cell, inlet gas composition and p_total in every gas
# cell, φ_s = 0 on GDL+CL (V_applied_SHE = 0 at zero overpotential) and φ_L = 0.
function make_u0_stage5_anchor(c_eq::Vector{Float64})
    u_liq = repeat(log.(c_eq), N_CL)
    return make_u0_stage5(make_u0_stage3(u_liq), V_RHE_to_SHE_offset)   # V_RHE = offset ⇒ V_c = 0
end

# φ_s-shift predictor fallback (§9.5.4): the collector boundary value moves by exactly dV.
function shift_phis!(u::Vector{Float64}, dV::Float64)
    for ig in 1:N_GDL; u[phi_s_gdl_idx(ig)] += dV; end
    for ix in 1:N_CL;  u[phi_s_cl_idx(ix)]  += dV; end
    return u
end

# Stage 5 current extractors: use LOCAL φ_s_cl, φ_L from the DOF vector (§17.4 note).
function compute_i_CO_s5(u::Vector{Float64}, cl_mesh)
    c = unpack_log_conc(u, cl_mesh.N); i = 0.0
    @inbounds for ix in 1:cl_mesh.N
        r = R_CO2R(c[5,ix], u[phi_s_cl_idx(ix)], u[phi_L_idx(ix)], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
function compute_i_H2_s5(u::Vector{Float64}, cl_mesh)
    c = unpack_log_conc(u, cl_mesh.N); i = 0.0
    @inbounds for ix in 1:cl_mesh.N
        r = R_HER(u[phi_s_cl_idx(ix)], u[phi_L_idx(ix)], c[1,ix], a_v_CL, n_e)
        i += n_e * F_const * r * cl_mesh.dx[ix]
    end
    return i
end
