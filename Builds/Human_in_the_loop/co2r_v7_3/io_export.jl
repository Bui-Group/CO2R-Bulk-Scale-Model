# ============================================================================
# io_export.jl — export converged sweeps to a summary CSV + a raw-binary
# profile file that plot_results.py reads with numpy.fromfile.
# ============================================================================
using Printf

struct SweepRecord
    V_RHE::Float64
    u::Vector{Float64}
    iters::Int
    resid::Float64
end

"""
    export_sweep(outdir, tag, ctx, recs)

Writes `<outdir>/summary_<tag>.csv` and `<outdir>/profiles_<tag>.bin`.

profiles_<tag>.bin layout (all little-endian native Float64 unless noted):
    Int64  nV, N_CL, N_GAS, stage
    Float64 x_cl[N_CL]      [µm], 0 = GDL face
    Float64 x_gas[N_GAS]    [µm], negative in the GDL
    per voltage:
      Float64 V_RHE
      Float64 liquid[9, N_CL]  (pH, CO2, OH, HCO3, CO3, Cs, H, R_CO2R, R_HER)
      Float64 gas[4, N_GAS]    (p_CO2 atm, p_CO atm, p_H2 atm, p_tot Pa)   [stage >= 3]
      Float64 phi_s[N_GAS], phi_L[N_CL]                                     [stage == 5]
"""
function export_sweep(outdir::String, tag::String, ctx::Ctx, recs::Vector{SweepRecord})
    mkpath(outdir)
    ly, cl, gm = ctx.ly, ctx.cl, ctx.gm
    N_CL = ly.N_CL; NG = gm.N; stage = ly.stage

    # ---- summary CSV ----
    open(joinpath(outdir, "summary_$(tag).csv"), "w") do io
        cols = "V_RHE,V_SHE,i_CO_mAcm2,i_H2_mAcm2,i_tot_mAcm2,FE_CO_pct,iters,resid_inf"
        stage == 5 && (cols *= ",dphi_L_mV,phi_s_drop_mV,charge_bal_rel,ionic_bal_rel")
        println(io, cols)
        for r in recs
            iCO, iH2, itot = compute_currents(r.u, r.V_RHE, ctx)
            fe = itot > 0 ? 100 * iCO / itot : 0.0
            line = @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.6f,%d,%.6e",
                            r.V_RHE, r.V_RHE - V_RHE_to_SHE_offset,
                            iCO/10, iH2/10, itot/10, fe, r.iters, r.resid)
            if stage == 5
                d = stage5_diagnostics(r.u, r.V_RHE, ctx)
                line *= @sprintf(",%.6f,%.6f,%.6e,%.6e",
                                 d.dphi_L_mV, d.phi_s_drop_mV, d.charge_bal_rel, d.ionic_bal_rel)
            end
            println(io, line)
        end
    end

    # ---- profiles binary ----
    open(joinpath(outdir, "profiles_$(tag).bin"), "w") do io
        write(io, Int64(length(recs))); write(io, Int64(N_CL))
        write(io, Int64(NG));           write(io, Int64(stage))
        write(io, collect(cl.x_centers .* 1e6))
        write(io, collect(gm.x_centers .* 1e6))
        for r in recs
            write(io, r.V_RHE)
            c = unpack_log_conc(r.u, ly)
            rco, rh = cell_rates(r.u, r.V_RHE, ctx)
            pH  = [-log10(max(c[1,ix]/1000, 1e-300)) for ix in 1:N_CL]
            cCs = [c_Cs_from_electroneutrality(view(c, :, ix)) for ix in 1:N_CL]
            write(io, pH)
            write(io, collect(c[5,:])); write(io, collect(c[2,:]))
            write(io, collect(c[3,:])); write(io, collect(c[4,:]))
            write(io, cCs);             write(io, collect(c[1,:]))
            write(io, rco);             write(io, rh)
            if stage >= 3
                pco2 = zeros(NG); pco = zeros(NG); ph2 = zeros(NG); ptot = zeros(NG)
                for g in 1:NG
                    w = gas_omega(r.u, ly, g); x = omega_to_x(w); p = gas_p(r.u, ly, g)
                    pco2[g] = x[1]*p/101325; pco[g] = x[2]*p/101325
                    ph2[g]  = x[3]*p/101325; ptot[g] = p
                end
                write(io, pco2); write(io, pco); write(io, ph2); write(io, ptot)
            end
            if stage >= 5
                phis = vcat([r.u[phi_s_gdl_idx(ly, ig)] for ig in 1:ly.N_GDL],
                            [r.u[phi_s_cl_idx(ly, ix)]  for ix in 1:N_CL])
                phiL = [r.u[phi_L_idx(ly, ix)] for ix in 1:N_CL]
                write(io, phis); write(io, phiL)
            end
        end
    end
    @printf("  exported %d voltages → %s/{summary,profiles}_%s.*\n",
            length(recs), basename(outdir), tag)
end

"""
    stage5_diagnostics(u, V_RHE, ctx)

§11 Stage-5 physicality checks 1/3/4/6 and the §3.1 Δφ_L sanity check.
"""
function stage5_diagnostics(u::Vector{Float64}, V_RHE::Float64, ctx::Ctx)
    ly, cl, gm = ctx.ly, ctx.cl, ctx.gm
    N = ly.N_CL
    V_applied_SHE = V_RHE - V_RHE_to_SHE_offset
    phiL = [u[phi_L_idx(ly, ix)] for ix in 1:N]
    phis_gdl1 = u[phi_s_gdl_idx(ly, 1)]
    phis_clN  = u[phi_s_cl_idx(ly, N)]

    rco, rh = cell_rates(u, V_RHE, ctx)
    i_far = n_e * F_const * sum((rco .+ rh) .* cl.dx)

    i_s1 = -sigma_s_eff_GDL * (phis_gdl1 - V_applied_SHE) / gm.dx_between[1]
    charge_rel = i_far > 0 ? abs(abs(i_far) - abs(i_s1)) / abs(i_far) : NaN

    # ionic current at the right (electrolyte) boundary, incl. migration
    c = unpack_log_conc(u, ly)
    s = 0.0
    for i in 1:5
        L_eff = D_eff[i] / k_MT[i]
        Ji = np_flux(c[i,N], ctx.c_eq[i], phiL[N], 0.0, D_eff[i], z_sp[i], L_eff)
        s += z_sp[i] * Ji
    end
    i_ionic_R = F_const * s          # J_Cs = 0 at this face by construction
    ionic_rel = i_far > 0 ? abs(abs(i_far) - abs(i_ionic_R)) / abs(i_far) : NaN

    return (dphi_L_mV     = 1000 * (phiL[1] - phiL[N]),
            phi_s_drop_mV = 1000 * (phis_clN - V_applied_SHE),
            charge_bal_rel = charge_rel,
            ionic_bal_rel  = ionic_rel,
            i_far = i_far, i_s1 = i_s1, i_ionic_R = i_ionic_R)
end
