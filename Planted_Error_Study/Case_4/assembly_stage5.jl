# assembly_stage5.jl — full Stage 5 residual: φ_L (NP migration + charge balance,
# incl. Cs⁺ migration) + φ_s (solid Ohm's law on GDL+CL). §3.1, §3.4, §11.
#
# DOF layout: [ liquid 5·N_CL ; gas 4·Ng ; φ_s (GDL then CL) Ng ; φ_L (CL) N_CL ],
# where Ng = N_GDL + N_CL. Liquid & gas blocks/indexing identical to Stage 4.

struct Stage5Layout
    N_CL::Int
    N_GDL::Int
    Ng::Int
    N_TOT::Int
end
function Stage5Layout(N_CL::Int, N_GDL::Int)
    Ng = N_GDL + N_CL
    Stage5Layout(N_CL, N_GDL, Ng, 5*N_CL + 4*Ng + Ng + N_CL)
end

@inline _phis_start(ly::Stage5Layout) = 5*ly.N_CL + 4*ly.Ng
@inline _phiL_start(ly::Stage5Layout) = _phis_start(ly) + ly.Ng
@inline phi_s_idx5(ly::Stage5Layout, s::Int) = _phis_start(ly) + s          # s=1..Ng (GDL then CL)
@inline phi_s_cl_idx5(ly::Stage5Layout, ix::Int) = _phis_start(ly) + ly.N_GDL + ix
@inline phi_L_idx5(ly::Stage5Layout, ix::Int) = _phiL_start(ly) + ix        # ix=1..N_CL

const L_eff_sh = [D_eff[i] / k_MT[i] for i in 1:5]      # Sherwood effective lengths

function residual_stage5!(F::AbstractVector{ElT}, u::AbstractVector{ElT},
                          V_cathode::Real, clm::Mesh, gm::GasMesh,
                          c_eq::Vector{Float64}, ly::Stage5Layout) where {ElT<:Real}
    N_CL = clm.N; N_GDL = gm.N_GDL; Ng = gm.Ng
    c = unpack_log_conc(u, N_CL)
    V_applied_SHE = V_cathode - V_RHE_to_SHE_offset

    φL    = ElT[u[phi_L_idx5(ly, ix)]    for ix in 1:N_CL]
    φs_cl = ElT[u[phi_s_cl_idx5(ly, ix)] for ix in 1:N_CL]

    # ── Darcy velocity per gas face ──
    uG = zeros(ElT, Ng + 1)
    for f in 2:Ng
        uG[f] = -(gm.kappa_face[f] / mu_gas) * (gas_p(u, N_CL, f) - gas_p(u, N_CL, f-1)) / gm.dx_cc[f]
    end
    uG[1] = -(gm.kappa_face[1] / mu_gas) * (gas_p(u, N_CL, 1) - p_total) / gm.dx_cc[1]

    # ── kinetics + phase transfer per CL cell (LOCAL φ_s, φ_L) ──
    Rco2r = zeros(ElT, N_CL); Rher = zeros(ElT, N_CL); Rpt = zeros(ElT, N_CL)
    for ix in 1:N_CL
        cix = @view c[:, ix]
        Rco2r[ix] = R_CO2R(cix[5], φs_cl[ix], φL[ix], cix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
        Rher[ix]  = R_HER(φs_cl[ix], φL[ix], cix[1], a_v_CL, n_e)
        p_co2 = pCO2_atm(u, N_CL, N_GDL + ix)
        Rpt[ix] = a_v_GL * k_GL() * (sechenov_H_CO2(cix) * p_co2 - cix[5])
    end

    # ── liquid NP fluxes (diffusion + migration) ──
    J = zeros(ElT, 5, N_CL + 1)
    p_co2_face = pCO2_atm(u, N_CL, N_GDL + 1)
    cCO2_bc = sechenov_H_CO2(@view c[:, 1]) * p_co2_face
    J[5, 1] = diffusion_flux(cCO2_bc, c[5, 1], D_eff[5], clm.dx_cc[1])       # ions no-flux at x=0
    for f in 2:N_CL, i in 1:5
        J[i, f] = np_flux(c[i, f-1], c[i, f], φL[f-1], φL[f], D_eff[i], z_charge[i], clm.dx_cc[f])
    end
    for i in 1:5
        J[i, N_CL+1] = np_flux(c[i, N_CL], c_eq[i], φL[N_CL], zero(ElT),
                               D_eff[i], z_charge[i], L_eff_sh[i])           # NP migration at right BC (§gotcha 12)
    end

    # ── Cs⁺ migration flux (for i_L only; §3.1 gotcha 9b) ──
    JCs = zeros(ElT, N_CL + 1)                # JCs[1]=JCs[N_CL+1]=0 (no flux at both CL faces)
    for f in 2:N_CL
        cCsL = cCs_from_neutrality(@view c[:, f-1])
        cCsR = cCs_from_neutrality(@view c[:, f])
        JCs[f] = np_flux(cCsL, cCsR, φL[f-1], φL[f], D_eff_Cs, z_Cs, clm.dx_cc[f])
    end

    # ── liquid species residuals ──
    for ix in 1:N_CL
        buf = buffer_sources(@view c[:, ix])
        for i in 1:5
            src = eps_I * buf[i]
            i == 2 && (src += 2*Rco2r[ix] + 2*Rher[ix])
            i == 5 && (src -= Rco2r[ix]; src += Rpt[ix])
            F[liquid_idx(ix, i)] = J[i, ix] - J[i, ix+1] + src * clm.dx[ix]
        end
    end

    # ── gas residual (Stefan-Maxwell + Darcy; sources from local-potential rates) ──
    Jg = zeros(ElT, 3, Ng + 1)
    let JJ = sm_flux(omega_inlet_tuple, gas_omega(u, N_CL, 1), p_total, gas_p(u, N_CL, 1),
                     gm.brugg[1], gm.dx_cc[1], uG[1])
        Jg[1,1] = JJ[1]; Jg[2,1] = JJ[2]; Jg[3,1] = JJ[3]
    end
    for f in 2:Ng
        JJ = sm_flux(gas_omega(u, N_CL, f-1), gas_omega(u, N_CL, f),
                     gas_p(u, N_CL, f-1), gas_p(u, N_CL, f), gm.brugg[f], gm.dx_cc[f], uG[f])
        Jg[1,f] = JJ[1]; Jg[2,f] = JJ[2]; Jg[3,f] = JJ[3]
    end
    for g in 1:Ng
        S1 = zero(ElT); S2 = zero(ElT); S3 = zero(ElT)
        if g > N_GDL
            ix = g - N_GDL
            S1 = -Rpt[ix]*M_gas[1]; S2 = Rco2r[ix]*M_gas[2]; S3 = Rher[ix]*M_gas[3]
        end
        F[gidx(N_CL, g, 1)] = Jg[1,g] - Jg[1,g+1] + S1 * gm.dx[g]
        F[gidx(N_CL, g, 2)] = Jg[2,g] - Jg[2,g+1] + S2 * gm.dx[g]
        F[gidx(N_CL, g, 3)] = Jg[3,g] - Jg[3,g+1] + S3 * gm.dx[g]
        ρuL = darcy_massflux(u, N_CL, g, g,   gm, uG, true)
        ρuR = darcy_massflux(u, N_CL, g, g+1, gm, uG, false)
        F[gidx(N_CL, g, 4)] = ρuL - ρuR + (S1 + S2 + S3) * gm.dx[g]
    end

    # ── φ_L residual: local ionic charge balance (§3.1) ──
    iL = zeros(ElT, N_CL + 1)
    for f in 1:N_CL+1
        s = zero(ElT)
        for i in 1:5; s += z_charge[i] * J[i, f]; end
        iL[f] = F_const * (s + z_Cs * JCs[f])
    end
    for ix in 1:N_CL
        i_far = n_e * F_const * (Rco2r[ix] + Rher[ix])
        F[phi_L_idx5(ly, ix)] = (iL[ix+1] - iL[ix]) + i_far * clm.dx[ix]
    end
    F[phi_L_idx5(ly, N_CL)] = φL[N_CL]                     # Dirichlet gauge override

    # ── φ_s residual: solid Ohm's law on GDL+CL (§3.4) ──
    is = zeros(ElT, Ng + 1)
    is[1] = -sigma_s_eff_GDL * (u[phi_s_idx5(ly, 1)] - V_applied_SHE) / gm.dx_cc[1]  # ghost BC
    for f in 2:Ng
        σL = (f-1) <= N_GDL ? sigma_s_eff_GDL : sigma_s_eff_CL
        σR = f     <= N_GDL ? sigma_s_eff_GDL : sigma_s_eff_CL
        σf = 2*σL*σR / (σL + σR)
        is[f] = -σf * (u[phi_s_idx5(ly, f)] - u[phi_s_idx5(ly, f-1)]) / gm.dx_cc[f]
    end
    # is[Ng+1] = 0 (Neumann, electrolyte side) — already zero
    for s in 1:Ng
        if s <= N_GDL
            F[phi_s_idx5(ly, s)] = is[s+1] - is[s]
        else
            ix = s - N_GDL
            i_far = n_e * F_const * (Rco2r[ix] + Rher[ix])
            F[phi_s_idx5(ly, s)] = (is[s+1] - is[s]) - i_far * clm.dx[ix]
        end
    end
    return F
end

# Build the V_c = 0 anchor initial guess (§9.5.5): equilibrium liquid, inlet gas,
# φ_s = 0 (= V_applied_SHE at V_c=0), φ_L = 0.
function anchor_guess(ly::Stage5Layout, c_eq::Vector{Float64})
    u = zeros(ly.N_TOT)
    for ix in 1:ly.N_CL
        u[5*(ix-1)+1:5*ix] .= log.(c_eq)
    end
    for g in 1:ly.Ng
        b = gas_base(ly.N_CL) + 4*(g-1)
        u[b+1] = omega_inlet[1]; u[b+2] = omega_inlet[2]; u[b+3] = omega_inlet[3]
        u[b+4] = p_total
    end
    # φ_s = 0, φ_L = 0 already
    return u
end

# Ionic current at the right (electrolyte) boundary and integrated faradaic current
# (§Stage 5 checks 4 & 6). Returns (i_far_total, i_ionic_R, i_s_collector) in A/m².
function stage5_currents(u::Vector{Float64}, V_cathode::Float64, clm::Mesh, gm::GasMesh,
                         c_eq::Vector{Float64}, ly::Stage5Layout)
    N_CL = clm.N; N_GDL = gm.N_GDL
    c = unpack_log_conc(u, N_CL)
    φL = [u[phi_L_idx5(ly, ix)] for ix in 1:N_CL]
    φs_cl = [u[phi_s_cl_idx5(ly, ix)] for ix in 1:N_CL]
    i_far = 0.0
    for ix in 1:N_CL
        r = R_CO2R(c[5,ix], φs_cl[ix], φL[ix], c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) +
            R_HER(φs_cl[ix], φL[ix], c[1,ix], a_v_CL, n_e)
        i_far += n_e * F_const * r * clm.dx[ix]
    end
    # ionic current at right boundary (NP flux incl. migration)
    iR = 0.0
    for i in 1:5
        Ji = np_flux(c[i,N_CL], c_eq[i], φL[N_CL], 0.0, D_eff[i], z_charge[i], L_eff_sh[i])
        iR += z_charge[i] * Ji
    end
    i_ionic_R = F_const * iR                              # Cs flux = 0 at right boundary
    # solid current at collector (ghost BC)
    i_s_coll = -sigma_s_eff_GDL * (u[phi_s_idx5(ly,1)] - (V_cathode - V_RHE_to_SHE_offset)) / gm.dx_cc[1]
    return i_far, i_ionic_R, i_s_coll
end


# ── §9.5.3 weight blocks for the Stage 5 layout:
#    liquid log-c | gas GDL | gas CL | φ_s GDL | φ_s CL | φ_L  (six contiguous blocks covering 1:N_TOT)
function dof_blocks(ly::Stage5Layout)
    blocks = dof_blocks(4, ly.N_CL, ly.N_GDL)
    push!(blocks, phi_s_idx5(ly, 1):phi_s_idx5(ly, ly.N_GDL))         # φ_s GDL (s = 1..N_GDL)
    push!(blocks, phi_s_cl_idx5(ly, 1):phi_s_cl_idx5(ly, ly.N_CL))    # φ_s CL
    push!(blocks, phi_L_idx5(ly, 1):phi_L_idx5(ly, ly.N_CL))          # φ_L
    @assert first(blocks[1]) == 1 && last(blocks[end]) == ly.N_TOT
    @assert all(last(blocks[k]) + 1 == first(blocks[k+1]) for k in 1:length(blocks)-1)
    return blocks
end

# ── φ_s-shift predictor fallback (§9.5.4): the collector boundary value moves by exactly dV,
#    so every φ_s DOF (GDL + CL) is shifted by dV. Used only when no tangent is available.
function shift_phis!(u::Vector{Float64}, dV::Float64, ly::Stage5Layout)
    for s in 1:ly.Ng; u[phi_s_idx5(ly, s)] += dV; end
    return u
end

nothing
