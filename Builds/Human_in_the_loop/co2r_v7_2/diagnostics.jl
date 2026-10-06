# =============================================================================
# diagnostics.jl — physicality checks (§11 per-stage, §12).
# All diagnostics are Float64-only post-processing on converged solutions.
# =============================================================================
using Printf

"""
    liquid_fluxes(u, V, ctx) -> J (5 x N_CL+1)

Recompute the liquid face fluxes of a converged solution using exactly the
same stencil as `full_residual!`.  Used by the mass-balance and ionic-current
checks (§12.6, §11 Stage 5 check 6).
"""
function liquid_fluxes(u::Vector{Float64}, V::Float64, ctx::ModelCtx)
    ly = ctx.ly; cl = ctx.cl; Nc = ly.N_CL
    c = unpack_log_conc(u, ly)
    phi_L = ly.has_pot ? unpack_phi_L_cl(u, ly) : zeros(Nc)
    J = zeros(5, Nc + 1)
    cCO2_bc = ly.has_gas ? sechenov_H_CO2(view(c, :, 1)) * pCO2_from_gas(u, ly, 1) :
                           c_CO2_henry_stage12
    # 2026-09-27 post-build change: no CO2(aq) flux at the GDL/CL face in Stage 5 (guide v7.1, Stage 5 note).
    J[5, 1] = ly.has_pot ? 0.0 : diffusion_flux(cCO2_bc, c[5, 1], D_eff[5], cl.dx_between[1])
    for f in 2:Nc, i in 1:5
        J[i, f] = ly.has_pot ?
            np_flux(c[i,f-1], c[i,f], phi_L[f-1], phi_L[f], D_eff[i], z_chg[i], cl.dx_between[f]) :
            diffusion_flux(c[i,f-1], c[i,f], D_eff[i], cl.dx_between[f])
    end
    if ly.stage == 1
        for i in 1:5
            J[i, Nc+1] = diffusion_flux(c[i,Nc], ctx.c_eq[i], D_eff[i], cl.dx_between[Nc+1])
        end
    elseif ly.has_pot
        for i in 1:5
            J[i, Nc+1] = np_flux(c[i,Nc], ctx.c_eq[i], phi_L[Nc], 0.0,
                                 D_eff[i], z_chg[i], D_eff[i]/k_MT[i])
        end
    else
        for i in 1:5
            J[i, Nc+1] = -k_MT[i] * (ctx.c_eq[i] - c[i,Nc])
        end
    end
    return J
end

"Volumetric CO2 phase-transfer rate per CL cell (zero before Stage 3)."
function phase_transfer_rates(u::Vector{Float64}, ctx::ModelCtx)
    ly = ctx.ly
    ly.has_gas || return zeros(ly.N_CL)
    c = unpack_log_conc(u, ly)
    return [a_v_GL * k_GL() * (sechenov_H_CO2(view(c,:,ix)) * pCO2_from_gas(u, ly, ix) - c[5,ix])
            for ix in 1:ly.N_CL]
end

"""
    co2_mass_balance(u, V, ctx) -> (residual, scale)

§12.6:  J_in(x=0) - J_out(x=L_CL) + int R_PT dx - int R_CO2R dx
        + int eps_I*S_buf_CO2 dx = 0.
The buffer term is part of the CO2(aq) budget and must be included.
"""
function co2_mass_balance(u::Vector{Float64}, V::Float64, ctx::ModelCtx)
    ly = ctx.ly; Nc = ly.N_CL
    J = liquid_fluxes(u, V, ctx)
    c = unpack_log_conc(u, ly)
    rc, _ = compute_rates(u, V, ctx)
    rpt = phase_transfer_rates(u, ctx)
    src = 0.0
    for ix in 1:Nc
        buf = buffer_sources(view(c, :, ix))
        src += (eps_I * buf[5] + rpt[ix] - rc[ix]) * ctx.cl.dx[ix]
    end
    bal = J[5,1] - J[5,Nc+1] + src
    scale = max(abs(J[5,1]), abs(J[5,Nc+1]), abs(src), 1e-30)
    return bal, scale
end

"""
    polarization_smoothness(Vs, iCO, iH2) -> (n_sign_CO, n_sign_H2)

§9.5.8 / §17.6: count sign changes in the first difference.  Requires exactly
1 for di_CO/dV (one rise, one rollover) and 0 for di_H2/dV (strictly
monotone).  A percentage-step test is meaningless near the i_CO peak.
"""
function polarization_smoothness(Vs::Vector{Float64}, iCO::Vector{Float64}, iH2::Vector{Float64})
    dCO = diff(iCO); dH2 = diff(iH2)
    cnt(d) = count(k -> sign(d[k]) != sign(d[k+1]) && d[k] != 0 && d[k+1] != 0,
                   1:(length(d)-1))
    return cnt(dCO), cnt(dH2)
end

"§11 Stage 2: Biot numbers Bi_i = k_MT_i * L_CL / D_eff_i."
biot_numbers() = [k_MT[i] * L_CL / D_eff[i] for i in 1:5]

"""
    stage_liquid_checks(Vs, sols, ctx)

The §11 Stage-1/2 and §12.3 liquid physicality checks, reported over the sweep.
"""
function stage_liquid_checks(Vs::Vector{Float64}, sols::Vector{Vector{Float64}}, ctx::ModelCtx)
    ly = ctx.ly; Nc = ly.N_CL
    println("\n--- Liquid-phase physicality checks (§11, §12.3) ---")
    kmid = argmin(abs.(Vs .- (-1.200)))
    for k in (1, kmid, length(Vs))
        u = sols[k]; c = unpack_log_conc(u, ly)
        pH = -log10.(c[1,:] ./ 1000.0)
        @printf("  V_RHE=%7.3f : pH %6.3f (x=0) -> %6.3f (x=L_CL) | c_CO2 %8.4f -> %8.4f | c_OH %10.4f -> %10.4f\n",
                Vs[k], pH[1], pH[end], c[5,1], c[5,end], c[2,1], c[2,end])
    end
    u = sols[end]; c = unpack_log_conc(u, ly); pH = -log10.(c[1,:] ./ 1000.0)
    @printf("  pH decreases toward the electrolyte (§12.3): %s  (pH[1]=%.3f > pH[N]=%.3f)\n",
            pH[1] > pH[end], pH[1], pH[end])
    @printf("  c_OH(L_CL) above bulk equilibrium %.4e (§11 Stage 2): %s (=%.4e)\n",
            ctx.c_eq[2], c[2,end] > ctx.c_eq[2], c[2,end])
    pHmin = minimum([minimum(-log10.(unpack_log_conc(s, ly)[1,:] ./ 1000)) for s in sols])
    pHmax = maximum([maximum(-log10.(unpack_log_conc(s, ly)[1,:] ./ 1000)) for s in sols])
    @printf("  pH range over the whole sweep: %.3f .. %.3f (§11 Stage 1 expects ~9-12 in operation)\n", pHmin, pHmax)

    mbmax = 0.0
    for (k, V) in enumerate(Vs)
        b, s = co2_mass_balance(sols[k], V, ctx)
        mbmax = max(mbmax, abs(b))
    end
    @printf("  §12.6 CO2 mass balance: max |residual| = %.3e mol/m2/s (tol 1e-6)  PASS=%s\n",
            mbmax, mbmax < 1e-6)
    return nothing
end

"Experimental comparison at the 5 §16.1 voltages, by interpolation on the sweep."
function compare_to_experiment(Vs::Vector{Float64}, iCO::Vector{Float64}, iH2::Vector{Float64})
    println("\n--- Comparison with Romiluyi et al. 2023 (§16.1) ---")
    println("   V_RHE    i_CO model / exp  (rel)      i_H2 model / exp  (rel)     FE_CO m/e")
    ord = sortperm(Vs)
    Vso = Vs[ord]; co = iCO[ord]; h2 = iH2[ord]
    lin(x, xs, ys) = begin
        j = searchsortedfirst(xs, x)
        j <= 1 && return ys[1]; j > length(xs) && return ys[end]
        t = (x - xs[j-1]) / (xs[j] - xs[j-1]); (1-t)*ys[j-1] + t*ys[j]
    end
    for (k, V) in enumerate(E_exp_RHE)
        mc = lin(V, Vso, co); mh = lin(V, Vso, h2)
        fem = 100 * mc / max(mc + mh, 1e-30)
        @printf("  %7.3f   %9.2f / %6.1f (%+6.1f%%)   %9.2f / %6.1f (%+6.1f%%)   %5.1f / %5.1f\n",
                V, mc, i_CO_exp[k], 100*(mc-i_CO_exp[k])/i_CO_exp[k],
                mh, i_H2_exp[k], 100*(mh-i_H2_exp[k])/i_H2_exp[k],
                fem, FE_CO_exp[k])
    end
end

"""
    sherwood_flux_check(u, V, ctx) -> max relative mismatch

§11 Stage 2: the species flux at the electrolyte face must equal
`k_MT_i * (c_i(L_CL) - c_eq_i)`.  The sign falls out of the concentration
difference — positive (leaving the CL) for OH- which is produced inside,
negative (reservoir resupplying) for CO2 which is depleted there.
"""
function sherwood_flux_check(u::Vector{Float64}, V::Float64, ctx::ModelCtx)
    ly = ctx.ly; Nc = ly.N_CL
    J = liquid_fluxes(u, V, ctx)
    c = unpack_log_conc(u, ly)
    worst = 0.0; rows = NTuple{4,Float64}[]
    for i in 1:5
        expect = k_MT[i] * (c[i, Nc] - ctx.c_eq[i])
        got    = J[i, Nc+1]
        rel    = abs(got - expect) / max(abs(expect), 1e-30)
        worst  = max(worst, rel)
        push!(rows, (Float64(i), got, expect, rel))
    end
    return worst, rows
end

"§11 Stage 2 electrolyte-face state and Biot numbers."
function stage2_checks(Vs::Vector{Float64}, sols::Vector{Vector{Float64}}, ctx::ModelCtx)
    println("\n--- Stage 2 checks (§11 Stage 2) ---")
    Bi = biot_numbers()
    println("  Biot numbers Bi_i = k_MT_i L_CL / D_eff_i  (want O(0.1-10)):")
    for i in 1:5
        @printf("    %-8s Bi = %8.3f   k_MT = %.4e m/s   D_eff = %.4e m2/s\n",
                SPECIES_NAMES[i], Bi[i], k_MT[i], D_eff[i])
    end
    @printf("  all in [0.1, 10]: %s\n", all(0.1 .<= Bi .<= 10.0))

    worst = 0.0
    for (k, V) in enumerate(Vs)
        w, _ = sherwood_flux_check(sols[k], V, ctx)
        worst = max(worst, w)
    end
    @printf("  electrolyte-face flux == k_MT*(c_local - c_eq): max rel. mismatch = %.3e\n", worst)

    _, rows = sherwood_flux_check(sols[end], Vs[end], ctx)
    @printf("  at V_RHE = %.3f, face fluxes [mol/m2/s]:\n", Vs[end])
    for r in rows
        @printf("    %-8s J = %+.4e   (sign %s)\n", SPECIES_NAMES[Int(r[1])], r[2],
                r[2] > 0 ? "leaving the CL" : "entering the CL")
    end

    cend = [unpack_log_conc(s, ctx.ly)[5, end] for s in sols]
    ohend = [unpack_log_conc(s, ctx.ly)[2, end] for s in sols]
    @printf("  c_CO2(L_CL): %.4f (mild) -> %.4f (most cathodic); min over sweep %.4f mol/m3\n",
            cend[1], cend[end], minimum(cend))
    @printf("  c_OH(L_CL) : %.4e (mild) -> %.4e ; bulk c_eq_OH = %.4e ; above bulk everywhere: %s\n",
            ohend[1], ohend[end], ctx.c_eq[2], all(ohend .> ctx.c_eq[2]))
    return nothing
end

"Gas-phase state helper: (omega 4-vector, x 4-vector, p) at combined gas cell m."
function gas_state_at(u::Vector{Float64}, ctx::ModelCtx, m::Int)
    w, p = gas_cell_state(u, ctx.ly, m)
    return (collect(w), collect(omega_to_x(w)), p)
end

"""
    stage3_checks(Vs, sols, ctx; pinned)

§11 Stage 3 physicality checks: omega_CO2 falls from the ~0.947 inlet value,
omega_CO and omega_H2 rise through the CL, and the CL CO2(aq) is consistent
with Henry's law against the local gas-phase p_CO2.  When `pinned` is true
(Stage 3) the pressure must be *exactly* uniform at p_total.
"""
function stage3_checks(Vs::Vector{Float64}, sols::Vector{Vector{Float64}},
                       ctx::ModelCtx; pinned::Bool = true)
    ly = ctx.ly; Ng = ly.N_GDL; Nc = ly.N_CL; Nt = Ng + Nc
    println("\n--- Stage 3 gas-phase checks (§11 Stage 3) ---")
    println("   V_RHE    w_CO2 inlet -> CL exit   x_CO2 in->out    w_CO exit    w_H2 exit")
    for k in (1, argmin(abs.(Vs .- (-1.200))), argmin(abs.(Vs .- (-1.650))), length(Vs))
        u = sols[k]
        w1, x1, _ = gas_state_at(u, ctx, 1)
        we, xe, _ = gas_state_at(u, ctx, Nt)
        @printf("  %7.3f   %8.5f -> %8.5f      %6.4f -> %6.4f   %10.3e   %10.3e\n",
                Vs[k], w1[1], we[1], x1[1], xe[1], we[2], we[3])
    end
    u = sols[end]
    wCO2 = [gas_state_at(u, ctx, m)[1][1] for m in 1:Nt]
    wCO  = [gas_state_at(u, ctx, m)[1][2] for m in 1:Nt]
    wH2  = [gas_state_at(u, ctx, m)[1][3] for m in 1:Nt]
    xCO2 = [gas_state_at(u, ctx, m)[2][1] for m in 1:Nt]
    pCO2 = [gas_state_at(u, ctx, m)[2][1] * gas_state_at(u, ctx, m)[3] for m in 1:Nt]
    @printf("  at V_RHE = %.3f: w_CO2 monotone decreasing across GDL+CL: %s\n",
            Vs[end], all(diff(wCO2) .<= 1e-14))
    @printf("                   x_CO2 (MOLE fraction) monotone decreasing:  %s (%.5f -> %.5f)\n",
            all(diff(xCO2) .<= 1e-14), xCO2[1], xCO2[Nt])
    @printf("                   p_CO2 monotone decreasing:                  %s (%.1f -> %.1f Pa)\n",
            all(diff(pCO2) .<= 1e-9), pCO2[1], pCO2[Nt])
    @printf("                   w_CO  increases through the CL: %s (%.3e -> %.3e)\n",
            wCO[Ng+1] < wCO[Nt], wCO[Ng+1], wCO[Nt])
    @printf("                   w_H2  increases through the CL: %s (%.3e -> %.3e)\n",
            wH2[Ng+1] < wH2[Nt], wH2[Ng+1], wH2[Nt])

    # pressure
    pall = vcat([[gas_state_at(s, ctx, m)[3] for m in 1:Nt] for s in sols]...)
    dpmax = maximum(abs.(pall .- p_total))
    if pinned
        @printf("  pressure pinned at p_total (§11 Stage 3): max |p - 101325| = %.3e Pa  PASS=%s\n",
                dpmax, dpmax < 1e-6)
    else
        dps = [maximum(v) - minimum(v) for v in
               [[gas_state_at(s, ctx, m)[3] for m in 1:Nt] for s in sols]]
        dpcl = [maximum([gas_state_at(s, ctx, m)[3] for m in Ng+1:Nt]) -
                minimum([gas_state_at(s, ctx, m)[3] for m in Ng+1:Nt]) for s in sols]
        @printf("  total pressure drop across GDL+CL: %.1f Pa (mild) -> %.1f Pa (cathodic)\n",
                dps[1], dps[end])
        @printf("  pressure drop across the CL alone : %.1f Pa (mild) -> %.1f Pa (cathodic)\n",
                dpcl[1], dpcl[end])
        return dps, dpcl
    end

    # Henry consistency at the gas-liquid interface
    println("  Henry consistency c_CO2(aq) vs H_eff*p_CO2 in the CL:")
    for k in (1, length(Vs))
        u = sols[k]; c = unpack_log_conc(u, ly)
        r1 = c[5,1]  / (sechenov_H_CO2(view(c,:,1))  * pCO2_from_gas(u, ly, 1))
        rN = c[5,Nc] / (sechenov_H_CO2(view(c,:,Nc)) * pCO2_from_gas(u, ly, Nc))
        @printf("    V_RHE=%7.3f : cell 1 ratio %.4f, cell N ratio %.4f  (1.0 = local Henry equilibrium)\n",
                Vs[k], r1, rN)
    end
    return nothing
end

"""
    gas_mass_balance(u, V, ctx) -> (inlet_rhou, net_source, rel_err)

§11 Stage 4: total Darcy mass flux at the GDL inlet must equal the net gas
mass source integrated over the CL.
"""
function gas_mass_balance(u::Vector{Float64}, V::Float64, ctx::ModelCtx)
    ly = ctx.ly; gg = ctx.gg
    rc, rh = compute_rates(u, V, ctx)
    rpt = phase_transfer_rates(u, ctx)
    S = 0.0
    for ix in 1:ly.N_CL
        S += (-rpt[ix]*M_gas[1] + rc[ix]*M_gas[2] + rh[ix]*M_gas[3]) * ctx.cl.dx[ix]
    end
    wR, pR = gas_cell_state(u, ly, 1)
    f1 = gas_face_flux(inlet_omega(Float64), p_total, wR, pR,
                       gg.Dface[1], gg.kappa[1], gg.dxc[1], ly.stage >= 4)
    rhou_in = f1[4]
    rel = abs(rhou_in + S) / max(abs(rhou_in), abs(S), 1e-30)
    return rhou_in, S, rel
end

"""
    stage4_checks(Vs, sols, ctx, iCO, iH2)

§11 Stage 4: Darcy mass conservation at the GDL inlet, the CL pressure drop,
and the split of the dissolving CO2 between CO2R and carbonate formation.
"""
function stage4_checks(Vs::Vector{Float64}, sols::Vector{Vector{Float64}}, ctx::ModelCtx,
                       iCO::Vector{Float64}, iH2::Vector{Float64})
    ly = ctx.ly; Ng = ly.N_GDL; Nc = ly.N_CL; Nt = Ng + Nc
    println("\n--- Stage 4 Darcy checks (§11 Stage 4) ---")

    worst = 0.0
    for (k, V) in enumerate(Vs)
        _, _, rel = gas_mass_balance(sols[k], V, ctx)
        worst = max(worst, rel)
    end
    @printf("  inlet Darcy mass flux == net CL gas mass source: max rel. error = %.3e\n", worst)

    println("\n  pressure drop and the gas mass budget:")
    println("     V_RHE   i_tot[A/cm2]   dp_CL[Pa]   dp_tot[Pa]   dp/p[%]   rhou_in[kg/m2/s]")
    idx = [argmin(abs.(Vs .- v)) for v in (-1.200, -1.650, -2.075, -2.500)]
    for k in idx
        u = sols[k]; V = Vs[k]
        ps = [gas_state_at(u, ctx, m)[3] for m in 1:Nt]
        dp_cl  = maximum(ps[Ng+1:Nt]) - minimum(ps[Ng+1:Nt])
        dp_tot = maximum(ps) - minimum(ps)
        rin, _, _ = gas_mass_balance(u, V, ctx)
        @printf("   %7.3f   %10.4f   %9.1f   %10.1f   %7.3f   %+.4e\n",
                V, (iCO[k]+iH2[k])/1000, dp_cl, dp_tot, 100*dp_tot/p_total, rin)
    end

    println("\n  where the dissolving CO2 goes (§11 Stage 4):")
    println("     V_RHE   int R_PT*M_CO2      i_CO/2F*M_CO      i_H2/2F*M_H2   CO2R share of R_PT")
    for k in idx
        u = sols[k]; V = Vs[k]
        rc, rh = compute_rates(u, V, ctx)
        rpt = phase_transfer_rates(u, ctx)
        mpt = sum(rpt .* ctx.cl.dx) * M_gas[1]
        mco = sum(rc  .* ctx.cl.dx) * M_gas[2]
        mh2 = sum(rh  .* ctx.cl.dx) * M_gas[3]
        share = 100 * sum(rc .* ctx.cl.dx) / max(sum(rpt .* ctx.cl.dx), 1e-30)
        @printf("   %7.3f      %.4e        %.4e        %.4e        %6.1f %%\n",
                V, mpt, mco, mh2, share)
    end
    return nothing
end

# ── Stage 5 diagnostics (§11 Stage 5 checks, §9.5.8 checklist) ───────────────
"""
    stage5_diagnostics(u, V, ctx) -> NamedTuple

Everything the §11 Stage-5 checks and the §9.5.8 checklist need from one
converged solution: the solid ohmic drop and its GDL/CL split, the liquid
potential drop across the CL, and the two independent charge-conservation
checks (solid current at the collector vs integrated faradaic current, and
ionic current at the electrolyte boundary vs the same).
"""
function stage5_diagnostics(u::Vector{Float64}, V::Float64, ctx::ModelCtx)
    ly = ctx.ly; Ng = ly.N_GDL; Nc = ly.N_CL
    V_applied_SHE = V - V_RHE_to_SHE_offset
    phi_s_gdl = unpack_phi_s_gdl(u, ly)
    phi_s_cl  = unpack_phi_s_cl(u, ly)
    phi_L     = unpack_phi_L_cl(u, ly)

    # §11 check 4: |i_s_face[1]| must equal the integrated faradaic current
    i_s1 = -sigma_s_eff_GDL * (phi_s_gdl[1] - V_applied_SHE) / ctx.gdl.dx_between[1]
    rc, rh = compute_rates(u, V, ctx)
    i_far = n_e * F_const * sum((rc .+ rh) .* ctx.cl.dx)
    d_solid = abs(abs(i_far) - abs(i_s1)) / max(abs(i_far), 1e-30)

    # §11 check 6: ionic current at the right boundary (NP flux, migration included)
    J = liquid_fluxes(u, V, ctx)
    i_ion_R = F_const * sum(z_chg[i] * J[i, Nc+1] for i in 1:5)
    d_ionic = abs(abs(i_far) - abs(i_ion_R)) / max(abs(i_far), 1e-30)

    # solid ohmic drop and its GDL/CL split (§11 checks 1-2)
    drop_tot = phi_s_cl[Nc] - V_applied_SHE
    drop_gdl = phi_s_gdl[Ng] - V_applied_SHE
    drop_cl  = drop_tot - drop_gdl

    dphiL = phi_L[1] - phi_L[Nc]
    monotone_L = all(diff(phi_L) .>= -1e-12) || all(diff(phi_L) .<= 1e-12)

    return (i_s1 = i_s1, i_far = i_far, d_solid = d_solid,
            i_ion_R = i_ion_R, d_ionic = d_ionic,
            drop_tot = drop_tot, drop_gdl = drop_gdl, drop_cl = drop_cl,
            gdl_share = 100 * drop_gdl / max(abs(drop_tot), 1e-30) * sign(drop_tot),
            dphiL = dphiL, phiL1 = phi_L[1], phiLmid = phi_L[cld(Nc,2)], phiLN = phi_L[Nc],
            monotone_L = monotone_L,
            cCs_max = maximum([cCs_from_electroneutrality(view(unpack_log_conc(u, ly), :, ix))
                               for ix in 1:Nc]))
end
