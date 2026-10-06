# run_stage4.jl — Stage 4: Darcy pressure finalized (§11 Stage 4)
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage4", "cache")
const PREFIX    = "s4_"

m = build_model(4)
print_kinetics_header(4, CACHE_DIR)
@printf("mesh: N_CL = %d, N_GDL = %d, DOFs = %d\n", m.cl.N, m.gdl.N, m.ly.N_TOT)
@printf("kappa_GDL = %.3e, kappa_CL = %.3e, kappa_interface = %.3e m²\n\n",
        kappa_GDL, kappa_CL, kappa_INT)

res_at_V!(F, u, V) = residual_stage34!(F, u, V, m.cl, m.gm, m.c_eq, m.ly, true)

u0 = u0_stage(m.ly, m.c_eq)
t0 = time()
Vs, Us, rows, plu = run_continuation(u0, copy(V_RHE_grid), trues(89), res_at_V!;
        blocks = weight_blocks(m.ly), n_log = 5*m.cl.N,
        cache_dir = CACHE_DIR, prefix = PREFIX, verbose = false)
@printf("wall time: %.1f s\n", time() - t0)
export_all("stage4", Vs, Us, m.ly, m.cl, m.gm, rows)

@printf("\n=== Stage 4 summary ===\n")
@printf("converged %d/%d, mean iters = %.2f, max ||F||inf = %.3e\n",
        length(Vs), 89, sum(r.iters for r in rows)/length(rows),
        maximum(r.Finf for r in rows))

# ── §11 Stage 4 physicality checks
println("\n-- pressure drop, Darcy mass balance, and rho*u conservation --")
let
    @printf("%9s %9s %12s %12s %14s %12s %12s\n",
            "V_RHE","i[A/cm2]","dp_CL[Pa]","dp_GDL[Pa]","rho_u_in","net_src","rel_err")
    for Vt in (-1.200, -1.400, -1.650, -2.075, -2.500)
        k = argmin(abs.(Vs .- Vt)); u = Us[k]
        c = unpack_log_conc(u, m.cl.N)
        V_SHE = Vs[k] - V_RHE_to_SHE_offset
        iCO, iH2 = compute_currents(u, V_SHE, m.cl)
        r1, r2 = rate_profiles(u, m.ly, m.cl, V_SHE)
        ps = [cell_gas(u, m.ly, g)[2] for g in 1:m.ly.Ng]
        dp_CL  = ps[m.ly.N_GDL+1] - ps[end]
        dp_GDL = p_total - ps[m.ly.N_GDL]
        # inlet Darcy mass flux (face 1, ghost at p_total)
        w1, p1 = cell_gas(u, m.ly, 1)
        wg = ntuple(j->omega_inlet[j], 4)
        uG_in = -(kappa_GDL/mu_gas)*(p1 - p_total)/m.gm.dxc[1]
        wf = ntuple(j->0.5*(wg[j]+w1[j]),4); pf = 0.5*(p_total+p1)
        rho_in = pf*Mn_from_omega(wf)/(R_gas*T)
        rhou_in = rho_in*uG_in
        net = 0.0
        for ix in 1:m.cl.N
            wgc,pgc = cell_gas(u, m.ly, m.ly.N_GDL+ix)
            rpt = a_v_GL*k_GL()*(sechenov_H_CO2(@view c[:,ix])*pCO2_atm(wgc,pgc) - c[5,ix])
            net += (-rpt*M_gas[1] + r1[ix]*M_gas[2] + r2[ix]*M_gas[3])*m.cl.dx[ix]
        end
        @printf("%9.3f %9.3f %12.1f %12.3f %14.4e %12.4e %12.2e\n",
                Vs[k], (iCO+iH2)/1e4, dp_CL, dp_GDL, rhou_in, net,
                abs(rhou_in + net)/max(abs(rhou_in),1e-30))
    end
    println("  (rel_err is |rho_u_in + net_source| / |rho_u_in|: inflow must balance the net sink)")
end

println("\n-- share of dissolving CO2 that goes to CO2R (§11 Stage 4) --")
let
    for Vt in (-1.200, -1.650, -2.075, -2.500)
        k = argmin(abs.(Vs .- Vt)); u = Us[k]
        c = unpack_log_conc(u, m.cl.N)
        r1, _ = rate_profiles(u, m.ly, m.cl, Vs[k]-V_RHE_to_SHE_offset)
        rpt_tot = 0.0; rco_tot = 0.0
        for ix in 1:m.cl.N
            wgc,pgc = cell_gas(u, m.ly, m.ly.N_GDL+ix)
            rpt_tot += a_v_GL*k_GL()*(sechenov_H_CO2(@view c[:,ix])*pCO2_atm(wgc,pgc) - c[5,ix])*m.cl.dx[ix]
            rco_tot += r1[ix]*m.cl.dx[ix]
        end
        @printf("V_RHE=%.3f  ∫R_PT=%.4e  ∫R_CO2R=%.4e  share=%.0f%%\n",
                Vs[k], rpt_tot, rco_tot, 100*rco_tot/rpt_tot)
    end
end

println("\n-- Stage 3 vs Stage 4 currents (Darcy should be a tiny correction) --")
for V in (-0.816, -1.205, -1.644, -2.079, -2.413)
    local k = argmin(abs.(Vs .- V))
    u4 = Us[k]; u3 = load_solution(joinpath(@__DIR__,"output_stage3","cache"), "s3_", Vs[k])
    V_SHE = Vs[k] - V_RHE_to_SHE_offset
    a = compute_currents(u4, V_SHE, m.cl); b = compute_currents(u3, V_SHE, m.cl)
    @printf("V_RHE=%.3f  S3: i_CO=%8.2f i_H2=%8.2f | S4: i_CO=%8.2f i_H2=%8.2f | dCO=%+.3f%% dH2=%+.3f%%\n",
            Vs[k], b[1]/10, b[2]/10, a[1]/10, a[2]/10,
            100*(a[1]-b[1])/b[1], 100*(a[2]-b[2])/b[2])
end
println("\nSTAGE 4 COMPLETE.")
