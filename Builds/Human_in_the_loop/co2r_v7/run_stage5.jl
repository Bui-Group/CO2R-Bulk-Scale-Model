# =====================================================================
#  run_stage5.jl — Stage 5: full electrochemistry.
#    phi_L : N_CL DOFs, per-cell ionic charge balance, NP migration in all
#            liquid fluxes incl. the right boundary, Cs+ migration in i_L,
#            Dirichlet gauge phi_L[N_CL] = 0 (residual OVERRIDE).
#    phi_s : N_GDL + N_CL DOFs, solid Ohm's law, Dirichlet V_applied_SHE at
#            the current collector as a GHOST-CELL BC (never an override).
#  Seeded ONCE at V_c = 0 from constructed physics, then tangent-predictor
#  continuation in -0.025 V steps.  NO cache seeding (§11 gotcha #0).
#  Uses the FITTED kinetics from run_fit.jl (§11 gotcha #14).
# =====================================================================
include("co2r_core.jl")
using Printf

const STAGE        = 5
const STAGE5_DIR   = joinpath(@__DIR__, "output_stage5")
const STAGE5_CACHE = joinpath(STAGE5_DIR, "cache")
save_stage5(u, V) = save_solution(STAGE5_CACHE, "s5", u, V)
load_stage5(V)    = load_solution(STAGE5_CACHE, "s5", V)

# ── §11 gotcha #14: load the fitted kinetics BEFORE anything else ────
fp = joinpath(@__DIR__, "fitted_params.jl")
isfile(fp) || error("fitted_params.jl not found — run run_fit.jl first (§11 gotcha #14)")
include(fp)
cache_banner("Stage 5 (FITTED kinetics)", STAGE5_CACHE)
@printf("Verify vs run_fit.jl output: i0=%.4f  lambda=%.5f  HER_pf=%.6e  alpha_c=%.6f\n\n",
        i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())

N_CL = N_CL_DEFAULT; N_GDL = N_GDL_DEFAULT
cl   = make_cl_mesh(N_CL); gdl = make_gdl_mesh(N_GDL); gg = build_gas_geom(gdl, cl)
ly   = Stage5Layout(N_CL, N_GDL)
c_eq, eq = equilibrium_vector()
@printf("Mesh: N_CL=%d, N_GDL=%d\n", N_CL, N_GDL)
@printf("DOFs: 5*%d + 4*(%d+%d) + %d + %d + %d = %d\n",
        N_CL, N_GDL, N_CL, N_GDL, N_CL, N_CL, ly.N_TOT)
@printf("sigma_s_eff: GDL %.2f  CL %.2f  interface(harmonic) %.2f S/m\n",
        sigma_s_eff_GDL, sigma_s_eff_CL, sigma_s_eff_int)
@printf("D_eff_Cs = %.4e m2/s (z_Cs=+1) -> Cs+ migration IS in i_L\n", D_eff_Cs)
@printf("dx_half for the phi_s ghost BC = %.4e m\n\n", gdl.dx_between[1])

# ── §9.5.5 anchor at V_c = 0, constructed from physics ───────────────
V_anchor_rhe = V_RHE_to_SHE_offset                       # V_c = 0 in SHE
Vrhe_pre  = [V_RHE_to_SHE_offset - k*DV_STEP for k in 1:32]   # V_c = -0.025 .. -0.800
Vrhe_chain = vcat(Vrhe_pre, V_RHE_grid)                       # then the 89 output points
@printf("Continuation: anchor V_c = 0.000 (V_RHE = %+.5f), then %d pre-ramp steps\n",
        V_anchor_rhe, length(Vrhe_pre))
@printf("              down to V_c = %+.5f, then the %d output voltages (%d steps total)\n\n",
        Vrhe_pre[end] - V_RHE_to_SHE_offset, length(V_RHE_grid), length(Vrhe_chain))

u0 = make_u0_stage5_anchor(c_eq, ly, V_anchor_rhe)       # never from a cache

records, plu, ctx = continuation_sweep(Val(STAGE); ly=ly, cl=cl, gdl=gdl, gg=gg,
    c_eq=c_eq, u0=u0, V_anchor_rhe=V_anchor_rhe, V_chain_rhe=Vrhe_chain,
    emit_set=Set(V_RHE_grid), cache_dir=STAGE5_CACHE, prefix="s5", verbose=true)

write_summary(STAGE5_DIR, records)
export_all_profiles(STAGE5_DIR, records, STAGE5_CACHE, "s5", cl, gdl, ly)

# ── Stage 5 diagnostics ──────────────────────────────────────────────
"Integrated faradaic current, boundary solid current, and boundary ionic current."
function s5_diag(u, V_rhe)
    c = unpack_log_conc(u, N_CL)
    V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
    phi_s_gdl = [u[phi_s_gdl_idx5(ly,ig)] for ig in 1:N_GDL]
    phi_s_cl  = [u[phi_s_cl_idx5(ly,ix)]  for ix in 1:N_CL]
    phi_L     = [u[phi_L_idx5(ly,ix)]     for ix in 1:N_CL]
    i_far = sum(n_e*F_const*(R_CO2R(c[5,ix], phi_s_cl[ix], phi_L[ix], c[1,ix],
                                    a_v_CL, lambda_MHC(), i0_CO2R(), n_e) +
                             R_HER(phi_s_cl[ix], phi_L[ix], c[1,ix], a_v_CL, n_e)) * cl.dx[ix]
                for ix in 1:N_CL)
    i_s1 = -sigma_s_eff_GDL*(phi_s_gdl[1] - V_applied_SHE)/gdl.dx_between[1]
    i_ion_R = F_const*sum(z_ion[i]*np_flux(c[i,N_CL], c_eq[i], phi_L[N_CL], 0.0,
                                           D_eff[i], z_ion[i], L_eff_bc[i]) for i in 1:5)
    return (i_far=i_far, i_s1=i_s1, i_ion_R=i_ion_R,
            phi_s_gdl=phi_s_gdl, phi_s_cl=phi_s_cl, phi_L=phi_L, V_applied_SHE=V_applied_SHE)
end

println("\n" * "="^72); println("STAGE 5 SUMMARY"); println("="^72)
@printf("Voltages converged : %d / %d\n", length(records), length(V_RHE_grid))
@printf("max |F|inf         : %.3e   (assertion tol %.0e)\n",
        maximum(r.resid for r in records), F_INF_TOL)
@printf("mean Newton iters  : %.2f  (max %d)\n",
        sum(r.iters for r in records)/length(records), maximum(r.iters for r in records))
@printf("sub-steps triggered: %d\n", sum(r.substeps for r in records))
iCO = [r.i_CO/10 for r in records]; iH2 = [r.i_H2/10 for r in records]
itot = iCO .+ iH2; FEs = 100 .* iCO ./ max.(itot,1e-30)
@printf("i_CO peak          : %.2f mA/cm2 at V_RHE = %.3f\n", maximum(iCO), records[argmax(iCO)].V_rhe)
@printf("FE_CO peak         : %.1f %% at V_RHE = %.3f\n", maximum(FEs), records[argmax(FEs)].V_rhe)

println("\nPer-voltage potential and charge diagnostics:")
@printf("%-9s %-10s %-11s %-11s %-11s %-11s %-11s %-11s\n", "V_RHE","i_tot","dphi_s tot",
        "GDL share","dphi_L","|di_s|/i","|di_ion|/i","phi_L mono")
for V in (-0.825, -1.200, -1.650, -2.075, -2.425, -2.500)
    u = load_stage5(V); d = s5_diag(u, V)
    dphis   = d.phi_s_cl[end] - d.V_applied_SHE
    dphis_g = d.phi_s_gdl[end] - d.V_applied_SHE
    dphiL   = d.phi_L[1] - d.phi_L[end]
    mono    = all(diff(d.phi_L) .>= -1e-12) || all(diff(d.phi_L) .<= 1e-12)
    @printf("%+.4f %9.1f %10.2f mV %8.1f %% %8.2f mV %10.2e %10.2e   %s\n",
            V, d.i_far/10, 1e3*dphis, 100*dphis_g/dphis, 1e3*dphiL,
            abs(abs(d.i_s1)-abs(d.i_far))/abs(d.i_far),
            abs(abs(d.i_far)-abs(d.i_ion_R))/abs(d.i_far), mono)
end

dphis_all = Float64[]; dphiL_all = Float64[]; dis_all = Float64[]; dion_all = Float64[]
for r in records
    d = s5_diag(load_stage5(r.V_rhe), r.V_rhe)
    push!(dphis_all, 1e3*(d.phi_s_cl[end]-d.V_applied_SHE)); push!(dphiL_all, 1e3*(d.phi_L[1]-d.phi_L[end]))
    push!(dis_all, abs(abs(d.i_s1)-abs(d.i_far))/abs(d.i_far))
    push!(dion_all, abs(abs(d.i_far)-abs(d.i_ion_R))/abs(d.i_far))
end
hi = itot .> 500.0
println("\n" * "-"^72)
@printf("Check 1  solid ohmic drop phi_s_CL[N]-V_applied  : %.2f .. %.2f mV (guide 5-80)\n",
        minimum(dphis_all), maximum(dphis_all))
@printf("Check 2  GDL share of the solid drop             : %.1f .. %.1f %% (guide 80-90)\n",
        minimum(100*(s5_diag(load_stage5(r.V_rhe),r.V_rhe).phi_s_gdl[end]-s5_diag(load_stage5(r.V_rhe),r.V_rhe).V_applied_SHE)/
                    (s5_diag(load_stage5(r.V_rhe),r.V_rhe).phi_s_cl[end]-s5_diag(load_stage5(r.V_rhe),r.V_rhe).V_applied_SHE) for r in records[10:end]),
        maximum(100*(s5_diag(load_stage5(r.V_rhe),r.V_rhe).phi_s_gdl[end]-s5_diag(load_stage5(r.V_rhe),r.V_rhe).V_applied_SHE)/
                    (s5_diag(load_stage5(r.V_rhe),r.V_rhe).phi_s_cl[end]-s5_diag(load_stage5(r.V_rhe),r.V_rhe).V_applied_SHE) for r in records[10:end]))
@printf("Check 3  |dphi_L| at i_tot > 500 mA/cm2          : %.2f .. %.2f mV (guide 5-25, ~15 typical)\n",
        minimum(abs.(dphiL_all[hi])), maximum(abs.(dphiL_all[hi])))
@printf("         sign of dphi_L (guide: negative)        : phi_L[1] = %+.3f mV at V_RHE=-2.075\n",
        1e3*s5_diag(load_stage5(-2.075),-2.075).phi_L[1])
@printf("Check 4  |i_s_face[1]| == i_far_total            : max rel err %.3e (guide <= 1e-6)\n",
        maximum(dis_all))
@printf("Check 6  ionic current conservation at x=L_CL    : max rel err %.3e (guide < 0.01)\n",
        maximum(dion_all))
s4 = joinpath(@__DIR__, "output_fit", "summary.csv")
if isfile(s4)
    d4 = Dict{Float64,NTuple{2,Float64}}()
    for ln in readlines(s4)[2:end]
        f = split(ln,","); d4[parse(Float64,f[1])] = (parse(Float64,f[7]), parse(Float64,f[8]))
    end
    drops = [100*(1 - (r.i_CO/10 + r.i_H2/10)/(d4[r.V_rhe][1]+d4[r.V_rhe][2])) for r in records if haskey(d4,r.V_rhe)]
    @printf("Check 5  i drop vs Stage-4-at-fitted-kinetics    : %.1f .. %.1f %% (guide 10-30 at high i)\n",
            minimum(drops), maximum(drops))
    for V in (-1.200,-1.650,-2.075,-2.425)
        k=findfirst(r->r.V_rhe==V, records); r=records[k]; b=d4[V]
        @printf("         V_RHE=%+.3f  i_CO %8.2f -> %8.2f (%+.1f%%)  i_H2 %8.2f -> %8.2f (%+.1f%%)\n",
                V, b[1], r.i_CO/10, 100*(r.i_CO/10-b[1])/b[1], b[2], r.i_H2/10, 100*(r.i_H2/10-b[2])/b[2])
    end
end
dsign = sum(diff(sign.(diff(iCO))) .!= 0)
@printf("Check 7  i_CO derivative sign changes            : %d (target <= 1)\n", dsign)
@printf("         i_H2 strictly monotone in |V|          : %s\n", all(diff(iH2) .> 0))
# smoothness is only meaningful past the exponential turn-on: restrict to
# i_CO above 10 % of peak (below that a 2e-3 -> 3e-3 mA/cm2 step reads as 100 %)
sel = findall(iCO[1:end-1] .> 0.10*maximum(iCO))
@printf("         max single-step |di_CO|/i_CO past onset: %.2f %% over %d voltages (spikes > 5%% = branch hop)\n",
        100*maximum(abs.(diff(iCO))[sel]./iCO[sel]), length(sel))
@printf("         max single-step |di_H2|/i_H2 past onset: %.2f %%\n",
        100*maximum(abs.(diff(iH2))[sel]./iH2[sel]))
@printf("         i_CO shape: rises to peak %.1f mA/cm2 at V_RHE=%.3f then falls to %.1f\n",
        maximum(iCO), records[argmax(iCO)].V_rhe, iCO[end])
println("="^72); flush(stdout)
