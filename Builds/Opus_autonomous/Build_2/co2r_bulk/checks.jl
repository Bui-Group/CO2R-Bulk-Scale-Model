# checks.jl — shared post-sweep physicality checks (§11, §12)

"Number of sign changes in the first difference of y (§9.5.8 smoothness test)."
function sign_changes(y::Vector{Float64})
    d = diff(y)
    n = 0
    s = 0
    for v in d
        v == 0 && continue
        sv = sign(v)
        s != 0 && sv != s && (n += 1)
        s = sv
    end
    return n
end

"""
    liquid_mass_balance(run, u, V) -> (co2_err, oh_err)

§12.6 — CO₂(aq) and OH⁻ integral balances from the converged state:
  J_in(x=0) − J_out(x=L_CL) + ∫S dx = 0     [mol/(m²·s)]
"""
function liquid_mass_balance(run::StageRun, u::Vector{Float64}, V::Float64)
    ly, cl = run.ly, run.cl
    N = ly.N_CL
    F = zeros(ly.N_TOT)
    res_at_V!(run, F, u, V)
    # the per-cell residual IS the local balance; summing over cells telescopes
    # the fluxes and leaves the global balance
    co2 = 0.0; oh = 0.0
    for ix in 1:N
        co2 += F[liquid_idx(ly, ix, 5)]
        oh  += F[liquid_idx(ly, ix, 2)]
    end
    return (co2, oh)
end

"Print the standard A-block physicality summary for a stage."
function print_physicality(run::StageRun, s, Vs::Vector{Float64}, us::Vector{Vector{Float64}})
    ly, cl = run.ly, run.cl
    N = ly.N_CL
    println("\n── Physicality checks ──")
    # pH range across all voltages
    pHmin = Inf; pHmax = -Inf; co2min = Inf
    for u in us
        c = unpack_log_conc(u, ly)
        for ix in 1:N
            ph = -log10(max(c[1,ix]/1000, 1e-30))
            pHmin = min(pHmin, ph); pHmax = max(pHmax, ph)
            co2min = min(co2min, c[5,ix])
        end
    end
    @printf("pH range over sweep : %.2f – %.2f\n", pHmin, pHmax)
    @printf("min c_CO2(aq)       : %.4f mol/m³\n", co2min)
    # deep-cathodic pH profile direction
    u = us[end]; c = unpack_log_conc(u, ly)
    @printf("pH(x=0) = %.2f  →  pH(x=L_CL) = %.2f   (must DECREASE toward electrolyte)\n",
            -log10(c[1,1]/1000), -log10(c[1,N]/1000))
    @printf("c_CO2: x=0 %.4f → x=L_CL %.4f mol/m³\n", c[5,1], c[5,N])
    @printf("c_OH : x=0 %.4e → x=L_CL %.4e mol/m³ (bulk eq %.3e)\n",
            c[2,1], c[2,N], run.c_eq[2])
    # currents
    sc_CO = sign_changes(s.iCO); sc_H2 = sign_changes(s.iH2)
    @printf("sign changes di_CO/dV = %d (target 1) ; di_H2/dV = %d (target 0)\n", sc_CO, sc_H2)
    kpk = argmax(s.FE_CO)
    @printf("peak FE_CO = %.1f %% at V_RHE = %.3f\n", s.FE_CO[kpk], Vs[kpk])
    @printf("i_tot at V_RHE=%.3f : %.2f mA/cm² (i_CO %.2f, i_H2 %.2f)\n",
            Vs[end], s.itot[end]/10, s.iCO[end]/10, s.iH2[end]/10)
    # mass balance at the most cathodic voltage
    (mco2, moh) = liquid_mass_balance(run, us[end], Vs[end])
    @printf("global CO₂(aq) balance = %.3e ; OH⁻ balance = %.3e mol/m²/s (tol 1e-6)\n", mco2, moh)
    # electrolyte-face fluxes (§11 Stage 2 check)
    if ly.stage >= 2
        JOH  = -k_MT[2]*(run.c_eq[2] - c[2,N])
        JCO2 = -k_MT[5]*(run.c_eq[5] - c[5,N])
        @printf("J_OH(L_CL) = %+.4e (should be >0, leaving) ; J_CO2 = %+.4e (should be <0)\n",
                JOH, JCO2)
        @printf("Biot numbers k_MT·L_CL/D_eff = %s (want O(0.1–10))\n",
                string(round.(k_MT .* L_CL ./ D_eff, sigdigits=3)))
    end
    return nothing
end

"Comparison of model currents against the 5 experimental points (§9.5.9 step 8)."
function compare_to_experiment(Vs::Vector{Float64}, iCO::Vector{Float64}, iH2::Vector{Float64})
    println("\n── vs experiment (Romiluyi et al. 2023) ──")
    println("  V_RHE    i_CO_mod  i_CO_exp   err%     i_H2_mod  i_H2_exp   err%")
    for (k, Ve) in enumerate(E_exp_RHE)
        j = argmin(abs.(Vs .- Ve))
        mCO = iCO[j]/10; mH2 = iH2[j]/10
        @printf("  %6.3f  %9.3f %9.3f  %+7.1f  %9.3f %9.3f  %+7.1f\n",
                Vs[j], mCO, i_CO_exp[k], 100*(mCO-i_CO_exp[k])/i_CO_exp[k],
                mH2, i_H2_exp[k], 100*(mH2-i_H2_exp[k])/i_H2_exp[k])
    end
end
