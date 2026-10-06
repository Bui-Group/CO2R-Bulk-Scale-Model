# run_stage4.jl — Stage 4: finalize the Darcy pressure equation + convective flux
include(joinpath(@__DIR__, "model.jl"))
using Printf

const N_CL  = N_CL_DEFAULT
const N_GDL = N_GDL_DEFAULT

run4 = make_stage_run(Val(4), N_CL, N_GDL)
print_kinetics_banner(4, run4.cache_dir)
@printf("mesh: N_CL=%d N_GDL=%d ; DOFs=%d\n", N_CL, N_GDL, run4.ly.N_TOT)
@printf("κ_GDL=%.3e  κ_CL=%.3e  κ_int=%.3e m²\n", kappa_GDL, kappa_CL,
        2*kappa_GDL*kappa_CL/(kappa_GDL+kappa_CL))

u0 = build_initial_guess(run4.ly, run4.cl, run4.gdl, run4.c_eq,
                         V_RHE_grid[1] - V_RHE_to_SHE_offset)
s3dir = joinpath(ROOT, "output_stage3", "cache")
fn = joinpath(s3dir, @sprintf("s3_V%.6f.bin", V_RHE_grid[1]))
if isfile(fn)
    uw = open(fn,"r") do f; n=read(f,Int64); read!(f, Vector{Float64}(undef,n)); end
    length(uw) == length(u0) && (u0 = uw)
    println("Warm-start: Stage-3 anchor, $(length(u0)) DOFs")
end

sw = run_sweep!(run4, u0, V_RHE_grid[1], V_RHE_grid[2:end], V_RHE_grid)
s  = export_stage(run4, sw.Vs, sw.us)

@printf("\n=== Stage 4 summary ===\n")
@printf("converged %d / %d voltages\n", length(sw.Vs), length(V_RHE_grid))
@printf("max ‖F‖∞ = %.3e ; mean Newton iters = %.2f ; wall = %.1f s\n",
        maximum(sw.resn), sum(sw.iters)/length(sw.iters), sw.wall)
print_physicality(run4, s, sw.Vs, sw.us)

# ── Stage-4 Darcy checks (§11 Stage 4) ──
println("\n── Stage 4 Darcy checks ──")
println("  i_tot[A/cm²]  Δp_CL[Pa]  ρu_inlet[kg/m²/s]  ΣS_mass·dx   mass-bal err   CO₂→CO₂R share")
NK = N_GDL + N_CL
ly = run4.ly
for k in (10, 30, 50, 70, 89)
    u = sw.us[k]; V = sw.Vs[k]
    c = unpack_log_conc(u, ly)
    Rco, Rher = compute_rates(u, ly, run4.cl, V)
    # phase transfer and mass sources per CL cell
    Smass = 0.0; Rpt_tot = 0.0
    for ix in 1:N_CL
        w = gas_omega(u, ly, N_GDL+ix); Mn = Mn_from_omega(w...)
        p_atm = omega_to_x(w[1],Mn,1)*gas_p(u,ly,N_GDL+ix)/101325
        H = sechenov_H_CO2(c[1,ix],c[2,ix],c[3,ix],c[4,ix])
        rpt = a_v_GL*k_GL()*(H*p_atm - c[5,ix])
        Rpt_tot += rpt*run4.cl.dx[ix]
        Smass += (-rpt*M_gas[1] + Rco[ix]*M_gas[2] + Rher[ix]*M_gas[3])*run4.cl.dx[ix]
    end
    # Darcy mass flux at the flow-field inlet face
    w1 = gas_omega(u, ly, 1); Mn1 = Mn_from_omega(w1...)
    wg = (omega_inlet[1],omega_inlet[2],omega_inlet[3],omega_inlet[4])
    Mng = Mn_from_omega(wg...)
    Mnf = Mn_from_omega(0.5.*(w1.+wg)...)
    pf = 0.5*(gas_p(u,ly,1)+p_total)
    rho_f = pf*Mnf/(R_gas*T)
    uG1 = -(kappa_GDL/mu_gas)*(gas_p(u,ly,1)-p_total)/run4.gdl.dxc[1]
    rhou1 = rho_f*uG1
    frac = (Rco[1]*0 + sum(Rco.*run4.cl.dx))/max(Rpt_tot,1e-30)
    @printf("  %10.4f  %9.1f  %+14.4e  %+12.4e  %11.3e  %8.1f %%\n",
            s.itot[k]/1e4, s.dp_CL[k], rhou1, Smass, abs(rhou1+Smass), 100*frac)
end
@printf("Stage 4 vs Stage 3 currents: max |Δi_tot|/i_tot = ")
s3 = joinpath(ROOT,"output_stage3","summary.csv")
if isfile(s3)
    import Printf
    ls = readlines(s3)[2:end]
    it3 = [parse(Float64, split(l,",")[5]) for l in ls]
    @printf("%.3e\n", maximum(abs.((s.itot./10 .- it3)./it3)))
else
    println("n/a")
end
compare_to_experiment(sw.Vs, s.iCO, s.iH2)
println("\nStage 4 complete — STOP for review.")
