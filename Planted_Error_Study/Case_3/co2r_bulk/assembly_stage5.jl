# assembly_stage5.jl — Stage 5 monolithic residual (§3.1, §3.4, §11).
# Adds φ_s (solid Ohm's law on GDL+CL) and φ_L (NP migration charge balance on CL) DOFs,
# switches kinetics to local (φ_s − φ_L). DOF layout:
#   [ liquid 5×N_CL ; gas 4×(N_GDL+N_CL) ; φ_s_GDL N_GDL ; φ_s_CL N_CL ; φ_L_CL N_CL ]

const GAS_END  = 5*N_CL + 4*(N_GDL + N_CL)
phi_s_gdl_idx(ig) = GAS_END + ig
phi_s_cl_idx(ix)  = GAS_END + N_GDL + ix
phi_L_idx(ix)     = GAS_END + N_GDL + N_CL + ix
const N_STAGE5 = GAS_END + N_GDL + 2*N_CL

# §9.5.3 weight blocks for each stage's DOF layout (iterate-derived norm of `newton_affine!`):
#   liquid log-c | gas GDL | gas CL | φ_s GDL | φ_s CL | φ_L   (later stages append blocks)
function dof_blocks(stage::Int)
    blocks = UnitRange{Int}[1:5*N_CL]
    if stage >= 3
        push!(blocks, (5*N_CL + 1):(5*N_CL + 4*N_GDL))          # gas_idx_g(g,·) for g ≤ N_GDL
        push!(blocks, (5*N_CL + 4*N_GDL + 1):GAS_END)           # gas_idx_g(g,·) for g > N_GDL
    end
    if stage >= 5
        push!(blocks, phi_s_gdl_idx(1):phi_s_gdl_idx(N_GDL))
        push!(blocks, phi_s_cl_idx(1):phi_s_cl_idx(N_CL))
        push!(blocks, phi_L_idx(1):phi_L_idx(N_CL))
    end
    return blocks
end

function unpack_phi_s(u::AbstractVector{ElT}) where {ElT<:Real}
    φs = zeros(ElT, N_GDL + N_CL)
    @inbounds for ig in 1:N_GDL; φs[ig] = u[phi_s_gdl_idx(ig)]; end
    @inbounds for ix in 1:N_CL;  φs[N_GDL+ix] = u[phi_s_cl_idx(ix)]; end
    return φs
end
unpack_phi_L(u::AbstractVector{ElT}) where {ElT<:Real} = ElT[u[phi_L_idx(ix)] for ix in 1:N_CL]

# Solid-phase electronic face currents i_s (§3.4). φs = [φ_s_gdl; φ_s_cl] (length n).
# Collector ghost BC at face 1; Neumann i_s=0 at electrolyte face n+1; harmonic-mean σ at interface.
function solid_face_currents(φs::AbstractVector{ElT}, V_applied::Real, gas_mesh) where {ElT<:Real}
    n = gas_mesh.n
    is = zeros(ElT, n+1)
    σcell(g) = g <= gas_mesh.N_GDL ? sigma_s_eff_GDL : sigma_s_eff_CL
    is[1] = -sigma_s_eff_GDL * (φs[1] - V_applied) / gas_mesh.dx_face_left   # ghost Dirichlet
    for f in 2:n
        σf = harm(σcell(f-1), σcell(f))
        is[f] = -σf * (φs[f] - φs[f-1]) / gas_mesh.dx_between[f-1]
    end
    is[n+1] = zero(ElT)                                                       # Neumann
    return is
end

"""
    residual_stage5!(F, u, V_cathode, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec)

Full Stage-5 residual: liquid NP (diffusion+migration) + SM/Darcy gas + φ_L charge balance
(incl. Cs⁺ migration) + φ_s solid Ohm's law. Kinetics use local (φ_s_cl − φ_L).
"""
function residual_stage5!(F::AbstractVector{ElT}, u::AbstractVector{ElT}, V_cathode::Real,
                          cl_mesh, gdl_mesh, gas_mesh, c_eq_vec) where {ElT<:Real}
    NCL = cl_mesh.N; NGDL = gdl_mesh.N; n = gas_mesh.n
    c  = unpack_log_conc(u, NCL)
    ω, p = unpack_gas(u, NCL, NGDL)
    φs = unpack_phi_s(u)
    φL = unpack_phi_L(u)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    # ── kinetics + phase transfer per CL cell (LOCAL φ_s, φ_L)
    Rco2r = zeros(ElT, NCL); Rher = zeros(ElT, NCL); Rpt = zeros(ElT, NCL)
    for ix in 1:NCL
        c_ix = @view c[:, ix]
        φs_loc = φs[NGDL + ix]; φL_loc = φL[ix]
        Rco2r[ix] = R_CO2R(c_ix[5], φs_loc, φL_loc, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rher[ix]  = R_HER(φs_loc, φL_loc, c_ix[1], a_v_CL, n_e)
        g = NGDL + ix; ωg = (ω[g,1],ω[g,2],ω[g,3],ω[g,4])
        pCO2 = omega_to_x(ωg)[1] * p[g] / 101325.0; H = sechenov_H_CO2(c_ix)
        Rpt[ix] = a_v_GL * k_GL() * (H*pCO2 - c_ix[5])
    end

    # ── liquid NP fluxes (diffusion + migration); CO₂ neutral so its migration term vanishes
    dxb = cl_mesh.dx_between; dxfl = cl_mesh.dx_face_left; dx = cl_mesh.dx
    g1 = NGDL + 1; ωg1 = (ω[g1,1],ω[g1,2],ω[g1,3],ω[g1,4])
    pCO2_1 = omega_to_x(ωg1)[1] * p[g1] / 101325.0
    cCO2bc = sechenov_H_CO2(@view c[:,1]) * pCO2_1
    Jl = zeros(ElT, 5, NCL+1)
    Jl[5,1] = diffusion_flux(cCO2bc, c[5,1], D_eff[5], dxfl)   # ions no-flux at GDL side (Jl[1:4,1]=0)
    for f in 2:NCL, i in 1:5
        Jl[i,f] = np_flux(c[i,f-1], c[i,f], φL[f-1], φL[f], D_eff[i], z_charge[i], dxb[f-1])
    end
    for i in 1:5
        Leff = D_eff[i] / k_MT[i]
        Jl[i,NCL+1] = np_flux(c[i,NCL], c_eq_vec[i], φL[NCL], zero(ElT), D_eff[i], z_charge[i], Leff)
    end

    # ── liquid species residuals
    for ix in 1:NCL
        buf = buffer_sources(@view c[:,ix])
        for i in 1:5
            src = eps_I*buf[i]
            if i == 2; src += 2*Rco2r[ix] + 2*Rher[ix]; end
            if i == 5; src -= Rco2r[ix]; src += Rpt[ix]; end
            F[liquid_idx(ix,i)] = Jl[i,ix] - Jl[i,ix+1] + src*dx[ix]
        end
    end

    # ── gas residual (Darcy on, continuity)
    Jg = zeros(ElT, 3, n+1); ρu = zeros(ElT, n+1)
    let κf = kappa_cell(1,NGDL), bf = gas_mesh.brugg[1], ωR = (ω[1,1],ω[1,2],ω[1,3],ω[1,4])
        J1,J2,J3,m = sm_face_flux(omega_inlet_tuple, p_total, ωR, p[1], gas_mesh.dx_face_left, bf, κf, true)
        Jg[1,1]=J1; Jg[2,1]=J2; Jg[3,1]=J3; ρu[1]=m
    end
    for f in 2:n
        gl=f-1; gr=f
        bf = harm(gas_mesh.brugg[gl], gas_mesh.brugg[gr]); κf = harm(kappa_cell(gl,NGDL), kappa_cell(gr,NGDL))
        ωL=(ω[gl,1],ω[gl,2],ω[gl,3],ω[gl,4]); ωR=(ω[gr,1],ω[gr,2],ω[gr,3],ω[gr,4])
        J1,J2,J3,m = sm_face_flux(ωL, p[gl], ωR, p[gr], gas_mesh.dx_between[f-1], bf, κf, true)
        Jg[1,f]=J1; Jg[2,f]=J2; Jg[3,f]=J3; ρu[f]=m
    end
    for g in 1:n
        dxg = gas_mesh.dx[g]
        if g <= NGDL
            S1=zero(ElT); S2=zero(ElT); S3=zero(ElT); Smass=zero(ElT)
        else
            ix = g-NGDL
            S1=-Rpt[ix]*M_gas[1]; S2=Rco2r[ix]*M_gas[2]; S3=Rher[ix]*M_gas[3]; Smass=S1+S2+S3
        end
        F[gas_idx_g(g,1,NCL,NGDL)] = Jg[1,g]-Jg[1,g+1]+S1*dxg
        F[gas_idx_g(g,2,NCL,NGDL)] = Jg[2,g]-Jg[2,g+1]+S2*dxg
        F[gas_idx_g(g,3,NCL,NGDL)] = Jg[3,g]-Jg[3,g+1]+S3*dxg
        F[gas_idx_g(g,4,NCL,NGDL)] = ρu[g]-ρu[g+1]+Smass*dxg
    end

    # ── φ_L charge balance (§3.1), incl. Cs⁺ migration in i_L
    cCs = ElT[c[2,ix]+c[3,ix]+2*c[4,ix]-c[1,ix]-C_fix*eps_I for ix in 1:NCL]
    JCs = zeros(ElT, NCL+1)                       # JCs[1]=JCs[NCL+1]=0 (no flux both CL faces)
    for f in 2:NCL
        JCs[f] = np_flux(cCs[f-1], cCs[f], φL[f-1], φL[f], D_eff_Cs, z_Cs, dxb[f-1])
    end
    iL = zeros(ElT, NCL+1)
    for f in 1:NCL+1
        s = zero(ElT)
        for i in 1:5; s += z_charge[i]*Jl[i,f]; end
        s += z_Cs*JCs[f]
        iL[f] = F_const * s
    end
    for ix in 1:NCL
        ifar = n_e*F_const*(Rco2r[ix]+Rher[ix])
        F[phi_L_idx(ix)] = (iL[ix+1] - iL[ix]) + ifar*dx[ix]
    end
    F[phi_L_idx(NCL)] = φL[NCL]                    # gauge override (φ_L=0 at electrolyte)

    # ── φ_s solid Ohm's law (§3.4)
    is = solid_face_currents(φs, V_applied_SHE, gas_mesh)
    for ig in 1:NGDL
        F[phi_s_gdl_idx(ig)] = is[ig+1] - is[ig]              # GDL: no faradaic
    end
    for ix in 1:NCL
        ifar = n_e*F_const*(Rco2r[ix]+Rher[ix])
        F[phi_s_cl_idx(ix)] = (is[NGDL+ix+1] - is[NGDL+ix]) - ifar*dx[ix]
    end
    return F
end

# ── warm-start builders ──
# Anchor guess at V_c=0 (§9.5.5 table): equilibrium liquid, inlet gas, φ_s=0, φ_L=0 —
# constructed from physics, never from a cache.
function anchor_guess_stage5()
    u = zeros(N_STAGE5)
    lnc = log.(equilibrium_c_vec(1000.0, 1000.0))
    for ix in 1:N_CL; u[liquid_idx(ix,1):liquid_idx(ix,5)] .= lnc; end
    for g in 1:(N_GDL+N_CL)
        base = g <= N_GDL ? 5*N_CL+4*(g-1) : 5*N_CL+4*N_GDL+4*(g-N_GDL-1)
        u[base+1]=omega_inlet[1]; u[base+2]=omega_inlet[2]; u[base+3]=omega_inlet[3]; u[base+4]=p_total
    end
    return u   # φ_s=0, φ_L=0
end

# Build Stage-5 warm start from a fit-cache (Stage-4 layout) solution + V (RHE): φ_s=V_applied_SHE, φ_L=0.
function make_u0_stage5(u_fit::Vector{Float64}, V_cathode::Float64)
    length(u_fit) == GAS_END || error("fit warm-start DOF mismatch: got $(length(u_fit)), expected $GAS_END")
    Vapp = V_cathode - V_RHE_to_SHE_offset
    u = zeros(N_STAGE5)
    u[1:GAS_END] .= u_fit
    for ig in 1:N_GDL; u[phi_s_gdl_idx(ig)] = Vapp; end
    for ix in 1:N_CL;  u[phi_s_cl_idx(ix)]  = Vapp; end
    # φ_L = 0 already
    return u
end

# φ_s-shift predictor fallback (§9.5.4): the collector boundary value moves by exactly dV, so
# shift every φ_s DOF by dV (used by `continuation_sweep!` when no tangent is available).
function shift_phis!(u::Vector{Float64}, dV::Float64)
    for ig in 1:N_GDL; u[phi_s_gdl_idx(ig)] += dV; end
    for ix in 1:N_CL;  u[phi_s_cl_idx(ix)]  += dV; end
    return u
end

# ── Stage 5 current extractors (local φ) ──
function compute_i_CO_s5(u, cl_mesh)
    c = unpack_log_conc(u, N_CL); φs = unpack_phi_s(u); φL = unpack_phi_L(u); i = 0.0
    for ix in 1:N_CL
        r = R_CO2R(c[5,ix], φs[N_GDL+ix], φL[ix], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), 2)
        i += 2*F_const*r*cl_mesh.dx[ix]
    end
    return i
end
function compute_i_H2_s5(u, cl_mesh)
    c = unpack_log_conc(u, N_CL); φs = unpack_phi_s(u); φL = unpack_phi_L(u); i = 0.0
    for ix in 1:N_CL
        r = R_HER(φs[N_GDL+ix], φL[ix], c[1,ix], a_v_CL, 2)
        i += 2*F_const*r*cl_mesh.dx[ix]
    end
    return i
end

# ── Stage 5 diagnostics: charge conservation |i_s(collector)|=i_far, and Δφ_L ──
function stage5_diagnostics(u, cl_mesh, gas_mesh, V)
    c = unpack_log_conc(u, N_CL); φs = unpack_phi_s(u); φL = unpack_phi_L(u)
    Vapp = V - V_RHE_to_SHE_offset
    ifar = 0.0
    for ix in 1:N_CL
        r = R_CO2R(c[5,ix], φs[N_GDL+ix], φL[ix], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) +
            R_HER(φs[N_GDL+ix], φL[ix], c[1,ix], a_v_CL, n_e)
        ifar += n_e*F_const*r*cl_mesh.dx[ix]
    end
    is1 = -sigma_s_eff_GDL*(φs[1] - Vapp)/gas_mesh.dx_face_left
    charge_err = abs(abs(is1) - abs(ifar)) / max(abs(ifar), 1e-30)
    dphiL = abs(φL[1] - φL[N_CL])
    solid_drop = φs[end] - Vapp     # φ_s rise from collector to catalyst
    return (ifar=ifar, is1=is1, charge_err=charge_err, dphiL=dphiL,
            phiL1=φL[1], phiLmid=φL[cld(N_CL,2)], phiLN=φL[N_CL], solid_drop=solid_drop)
end
