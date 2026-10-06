# run_stage2.jl — Stage 2: + Sherwood-Reynolds Neumann BC at the electrolyte side
include(joinpath(@__DIR__, "model.jl"))
using Printf

const N_CL  = N_CL_DEFAULT
const N_GDL = N_GDL_DEFAULT

run2 = make_stage_run(Val(2), N_CL, N_GDL)
print_kinetics_banner(2, run2.cache_dir)
@printf("mesh: N_CL=%d N_GDL=%d ; DOFs=%d\n", N_CL, N_GDL, run2.ly.N_TOT)
@printf("k_MT [m/s] = %s\n", string(round.(k_MT, sigdigits=4)))
@printf("Biot k_MT·L_CL/D_eff = %s\n", string(round.(k_MT .* L_CL ./ D_eff, sigdigits=3)))

# self-anchoring at −0.300 V RHE (§9.5.5); the Stage-1 cache is an optional
# warm-start convenience for the anchor only — never a per-voltage seed.
u0 = build_initial_guess(run2.ly, run2.cl, run2.gdl, run2.c_eq,
                         V_RHE_grid[1] - V_RHE_to_SHE_offset)
s1dir = joinpath(ROOT, "output_stage1", "cache")
if isdir(s1dir) && !isempty(list_cache(s1dir, "s1"))
    fn = joinpath(s1dir, @sprintf("s1_V%.6f.bin", V_RHE_grid[1]))
    if isfile(fn)
        uw = open(fn,"r") do f; n=read(f,Int64); read!(f, Vector{Float64}(undef,n)); end
    length(uw) == length(u0) && (u0 = uw)
        println("Warm-start: Stage-1 anchor, $(length(u0)) DOFs")
    end
end

sw = run_sweep!(run2, u0, V_RHE_grid[1], V_RHE_grid[2:end], V_RHE_grid)
s  = export_stage(run2, sw.Vs, sw.us)

@printf("\n=== Stage 2 summary ===\n")
@printf("converged %d / %d voltages, V_RHE ∈ [%.3f, %.3f]\n",
        length(sw.Vs), length(V_RHE_grid), sw.Vs[1], sw.Vs[end])
@printf("max ‖F‖∞ = %.3e ; mean Newton iters = %.2f ; wall = %.1f s\n",
        maximum(sw.resn), sum(sw.iters)/length(sw.iters), sw.wall)
print_physicality(run2, s, sw.Vs, sw.us)
compare_to_experiment(sw.Vs, s.iCO, s.iH2)
println("\nStage 2 complete — STOP for review.")
