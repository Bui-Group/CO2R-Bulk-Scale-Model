# §11 Stage 5 check 5: the measured Stage-4 -> Stage-5 shifts at V_RHE = -2.425,
# plus the §3.1 / §9.4-item-8 c_Cs figures.  Both stages at the FITTED kinetics:
# Stage 4 from output_fit/cache (the §17.8 refit sweep), Stage 5 from its own cache.
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
include(joinpath(@__DIR__, "fitted_kinetics.jl"))
using Printf

ctx4 = make_ctx(4); ctx5 = make_ctx(5)
V = -2.425
u4 = load_solution(joinpath(@__DIR__, "output_fit", "cache"), "fit_", V)
u5 = load_solution(cache_dir_for(5), cache_prefix_for(5), V)
(u4 === nothing || u5 === nothing) && error("missing cache at V = $V")

c4 = unpack_log_conc(u4, ctx4.ly); c5 = unpack_log_conc(u5, ctx5.ly)
cs4 = cCs_from_electroneutrality(view(c4,:,1)); cs5 = cCs_from_electroneutrality(view(c5,:,1))
H4 = sechenov_H_CO2(view(c4,:,1));              H5 = sechenov_H_CO2(view(c5,:,1))
oh4 = sum(c4[2,:] .* ctx4.cl.dx)/L_CL;          oh5 = sum(c5[2,:] .* ctx5.cl.dx)/L_CL
i4 = compute_currents(u4, V, ctx4); i5 = compute_currents(u5, V, ctx5)

println("="^96)
@printf("§11 Stage 5 check 5 — Stage 4 (fitted) -> Stage 5 at V_RHE = %.3f, CL cell 1\n", V)
println("="^96)
println("  quantity                      Stage 4      Stage 5      ratio     guide's measured values")
@printf("  c_CO3   [mol/m3]           %10.1f   %10.1f   %6.3f     4590 -> 2845  (-38%%)\n", c4[4,1], c5[4,1], c5[4,1]/c4[4,1])
@printf("  c_Cs    [mol/m3]           %10.1f   %10.1f   %6.3f     9369 -> 5537  (-41%%)\n", cs4, cs5, cs5/cs4)
@printf("  c_OH (CL-averaged)         %10.1f   %10.1f   %6.3f      410 ->  197  (-52%%)\n", oh4, oh5, oh5/oh4)
@printf("  H_eff   [mol/(m3 atm)]     %10.3f   %10.3f   %6.3f     1.66 -> 4.33  (2.6x)\n", H4, H5, H5/H4)
@printf("  c_CO2(aq) [mol/m3]         %10.3f   %10.3f   %6.3f     1.10 -> 3.26  (2.9x)\n", c4[5,1], c5[5,1], c5[5,1]/c4[5,1])
@printf("  i_CO    [mA/cm2]           %10.2f   %10.2f   %6.3f       86 ->  164  (1.90x plateau rise)\n", i4[1]/10, i5[1]/10, i5[1]/i4[1])
@printf("  i_H2    [mA/cm2]           %10.2f   %10.2f   %6.3f     falls 3-10%% (%.1f%%)\n", i4[2]/10, i5[2]/10, i5[2]/i4[2], 100*(i5[2]/i4[2]-1))

println("\n§3.1 / §9.4 item 8 — max c_Cs in the reaction zone across the Stage-5 sweep:")
for Vq in (-1.200, -2.500)
    u = load_solution(cache_dir_for(5), cache_prefix_for(5), Vq)
    c = unpack_log_conc(u, ctx5.ly)
    m = maximum([cCs_from_electroneutrality(view(c,:,ix)) for ix in 1:ctx5.ly.N_CL])
    @printf("  V_RHE = %.3f : c_Cs = %7.0f mol/m3   (guide: %s)\n", Vq, m,
            Vq == -1.200 ? "2045" : "6027")
end
u4b = load_solution(joinpath(@__DIR__, "output_fit","cache"), "fit_", -2.425)
c4b = unpack_log_conc(u4b, ctx4.ly)
m4 = maximum([cCs_from_electroneutrality(view(c4b,:,ix)) for ix in 1:ctx4.ly.N_CL])
u5b = load_solution(cache_dir_for(5), cache_prefix_for(5), -2.425)
c5b = unpack_log_conc(u5b, ctx5.ly)
m5 = maximum([cCs_from_electroneutrality(view(c5b,:,ix)) for ix in 1:ctx5.ly.N_CL])
@printf("  Stage 3-4 runs ~1.7x higher at V_RHE = -2.425: %6.0f vs %6.0f (ratio %.2f)  guide: 9369 vs 5537 (1.69)\n",
        m4, m5, m4/m5)

println("\n--- p_CO2 monotonicity detail at V_RHE = -2.500 (Stage 5) ---")
u = load_solution(cache_dir_for(5), cache_prefix_for(5), -2.500)
Nt = ctx5.ly.N_GDL + ctx5.ly.N_CL
pC = [gas_state_at(u, ctx5, m)[2][1] * gas_state_at(u, ctx5, m)[3] for m in 1:Nt]
d = diff(pC); bad = findall(d .> 0)
@printf("  p_CO2: %.1f -> %.1f Pa overall; %d of %d face-differences are positive\n",
        pC[1], pC[Nt], length(bad), length(d))
if !isempty(bad)
    @printf("  positive steps occur at cell indices %s (GDL/CL interface is at %d)\n",
            string(bad[1:min(end,8)]), ctx5.ly.N_GDL)
    @printf("  largest positive step: %+.4f Pa  (%.2e of the total %.1f Pa span)\n",
            maximum(d[bad]), maximum(d[bad])/(pC[1]-pC[Nt]), pC[1]-pC[Nt])
end
