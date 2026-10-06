# run_stage5.jl — Stage 5: phi_L (NP migration) + phi_s (solid Ohm's law)
# Uses the FITTED kinetics from run_fit.jl.  Seeded ONCE at V_c = 0 from a
# physics-constructed anchor, then tangent-predictor continuation (§9.5.5).
# NO cache seeding (§11 Stage 5 gotcha #0).
include("model.jl")
include("checks.jl")
using Printf

const STAGE_TAG = "stage5"
const PREFIX    = "s5_"

# ── §11 Stage 5 gotcha #14: load fitted kinetics BEFORE anything else
const FITFILE = joinpath(@__DIR__, "fitted_params.txt")
isfile(FITFILE) || error("fitted_params.txt not found — run run_fit.jl first")
let v = parse.(Float64, readlines(FITFILE))
    set_i0_CO2R!(v[1]); set_lambda_MHC!(v[2])
    set_HER_prefactor!(v[3]); set_alpha_c_HER!(v[4])
end

print_kinetics_header("Stage 5 — phi_L + phi_s with FITTED kinetics", STAGE_TAG)

N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
dm = make_domain(N_CL, N_GDL)
ly = Layout(5, N_CL, N_GDL)
@printf("Mesh: N_CL=%d, N_GDL=%d  ->  %d DOFs (expect %d)\n",
        N_CL, N_GDL, ly.N_TOT, 5*N_CL + 4*(N_GDL+N_CL) + N_GDL + N_CL + N_CL)
ctx = make_ctx(ly, dm)

# ── §9.5.5 anchor at V_c = 0 (SHE frame), built from physics
# V_c is V vs SHE; residual! takes V_cathode in RHE@pH8, so V_RHE = V_c + offset.
V_c_anchor = 0.0
V_RHE_anchor = V_c_anchor + V_RHE_to_SHE_offset
u0 = make_u0(ly, dm; V_applied_SHE = V_c_anchor)     # phi_s = 0, phi_L = 0

# pre-anchor ramp V_c = 0 -> first output point in -0.025 V steps, then the 89
V_c_first_out = V_SHE_grid[1]                        # = -0.81305
V_c_ramp = collect(0.0:-0.025:V_c_first_out)
if abs(V_c_ramp[end] - V_c_first_out) > 1e-12
    push!(V_c_ramp, V_c_first_out)
end
V_c_chain = vcat(V_c_ramp, V_SHE_grid[2:end])
Vs   = V_c_chain .+ V_RHE_to_SHE_offset              # to the RHE convention
emit = vcat(fill(false, length(V_c_ramp)-1), fill(true, 89))
@printf("Continuation chain: %d pre-anchor ramp points + %d output points = %d total\n",
        length(V_c_ramp), 89, length(Vs))
@printf("  anchor at V_c = %.4f V SHE (V_RHE = %+.4f)\n", V_c_anchor, V_RHE_anchor)

let Fw = zeros(ly.N_TOT)
    t = @elapsed ctx.res_at_V!(Fw, u0, V_RHE_anchor)
    @printf("JIT warm-up: residual %.3f s, ||F||inf at the anchor guess = %.3e\n",
            t, maximum(abs.(Fw)))
    t = @elapsed ctx.jacfun!(ctx.Jdense, u0)
    @printf("JIT warm-up: first AD Jacobian %.3f s\n", t); flush(stdout)
end

t = @elapsed Vout, Uout, st = continuation_sweep(ctx, u0, Vs; emit=emit,
        stage_tag=STAGE_TAG, prefix=PREFIX, quiet=true)

println()
@printf("Sweep complete: %d/89 output voltages converged in %.2f s\n", length(Vout), t)
@printf("  Newton iterations/step: mean %.2f, max %d\n",
        sum(st.iters)/length(st.iters), maximum(st.iters))
@printf("  max ||F||inf over the emitted voltages: %.3e (assertion tol %.0e)\n",
        maximum(st.resids), F_INF_TOL)
@printf("  sub-step recoveries fired: %d\n", st.n_substep)

println("\n--- §11 Stage 5 physicality checks ---")
res = report_polarization(Vout, Uout, ly, dm)
report_liquid(Vout, Uout, ly, dm)
mb = [abs(co2_mass_balance(Uout[k], ly, dm, Vout[k])) for k in 1:length(Vout)]
@printf("  CO2 mass-balance residual: max %.3e mol/m2/s (tol 1e-6)\n", maximum(mb))

# ── Stage-5 potential diagnostics
println("\n--- Stage 5 potential / charge-conservation diagnostics ---")
@printf("  %8s %9s %11s %11s %11s %11s %11s\n",
        "V_RHE","i_tot","dphi_L[mV]","phis_rise","GDL frac","|di|/i solid","ionic relerr")
worst_solid = 0.0; worst_ionic = 0.0
dphiL_all = Float64[]; itot_all = Float64[]
for k in 1:length(Vout)
    u = Uout[k]; V = Vout[k]
    V_SHE = V - V_RHE_to_SHE_offset
    Rco, Rher, c = compute_rates(u, ly, V_SHE)
    phi_s = [u[phi_s_idx(ly,kk)] for kk in 1:ly.N_G]
    phi_L = [u[phi_L_idx(ly,ix)] for ix in 1:ly.N_CL]
    iCO,iH2,it = compute_currents(u, ly, dm, V)
    i_far = sum(n_e*F_const*(Rco[ix]+Rher[ix])*dm.cl.dx[ix] for ix in 1:ly.N_CL)
    # solid charge conservation at the current collector (ghost-cell face)
    is1 = -sigma_s_eff_GDL*(phi_s[1] - V_SHE)/dm.dxcc_g[1]
    d_solid = abs(abs(i_far) - abs(is1))/max(abs(i_far),1e-30)
    # ionic current conservation at the right boundary (NP flux WITH migration)
    i_ionic_R = 0.0
    for i in 1:5
        L_eff = D_eff[i]/k_MT[i]
        Ji = np_flux(c[i,ly.N_CL], dm.c_eq[i], phi_L[ly.N_CL], 0.0, D_eff[i], z_sp[i], L_eff)
        i_ionic_R += F_const*z_sp[i]*Ji
    end
    d_ionic = abs(abs(i_far) - abs(i_ionic_R))/max(abs(i_far),1e-30)
    dphiL = (phi_L[1] - phi_L[ly.N_CL])*1e3
    phis_rise = (phi_s[ly.N_G] - V_SHE)*1e3
    gdl_frac = abs(phi_s[ly.N_GDL] - V_SHE)/max(abs(phi_s[ly.N_G]-V_SHE),1e-30)
    push!(dphiL_all, dphiL); push!(itot_all, it/10)
    global worst_solid = max(worst_solid, d_solid)
    global worst_ionic = max(worst_ionic, d_ionic)
    if k % 11 == 1 || k == length(Vout)
        @printf("  %8.3f %9.2f %11.3f %11.3f %11.3f %11.2e %11.2e\n",
                V, it/10, dphiL, phis_rise, gdl_frac, d_solid, d_ionic)
    end
end
@printf("  worst solid charge-conservation |di|/i : %.3e (target <= 1e-6)\n", worst_solid)
@printf("  worst ionic boundary rel error        : %.3e (target < 0.01)\n", worst_ionic)

# §9.5.8 dphi_L diagnostic at high current
hi = findall(itot_all .> 500.0)
if !isempty(hi)
    mags = abs.(dphiL_all[hi])
    @printf("  |dphi_L| at i_tot > 500 mA/cm2: %.2f - %.2f mV (target 5-25 mV)\n",
            minimum(mags), maximum(mags))
    @printf("  sign of dphi_L there: %s (expect negative: phi_L[1] < phi_L[N_CL] = 0)\n",
            all(dphiL_all[hi] .< 0) ? "all negative" : "MIXED - sign bug")
end
@printf("  phi_L sign consistent across all voltages: %s\n",
        all(dphiL_all .<= 1e-12) || all(dphiL_all .>= -1e-12))

println()
compare_to_experiment(Vout, Uout, ly, dm)

export_polarization(STAGE_TAG, Vout, Uout, ly, dm)
export_profiles(STAGE_TAG, Vout, Uout, ly, dm)
println("\nExported polarization.csv and profiles.bin to output_stage5/")
println("STAGE 5 COMPLETE — STOP GATE")
