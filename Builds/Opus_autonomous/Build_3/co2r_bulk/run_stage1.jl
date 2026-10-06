# run_stage1.jl — Stage 1: liquid NP diffusion only, Dirichlet BCs (§11 Stage 1)
include("common.jl")

const CACHE_DIR = joinpath(@__DIR__, "output_stage1", "cache")
const PREFIX    = "s1_"

m = build_model(1)
print_kinetics_header(1, CACHE_DIR)
@printf("mesh: N_CL = %d, DOFs = %d\n\n", m.cl.N, m.ly.N_TOT)

res_at_V!(F, u, V) = residual_stage12!(F, u, V, m.cl, m.c_eq, true)

u0 = u0_stage(m.ly, m.c_eq)
chain = copy(V_RHE_grid)
emit  = trues(length(chain))

t0 = time()
Vs, Us, rows, plu = run_continuation(u0, chain, emit, res_at_V!;
        blocks = weight_blocks(m.ly), n_log = 5*m.cl.N,
        cache_dir = CACHE_DIR, prefix = PREFIX, verbose = true)
@printf("\nwall time: %.1f s\n", time() - t0)

export_all("stage1", Vs, Us, m.ly, m.cl, m.gm, rows)

# ── summary + physicality checks (§11 Stage 1)
maxF = maximum(r.Finf for r in rows)
@printf("\n=== Stage 1 summary ===\n")
@printf("converged %d/%d voltages, V_RHE ∈ [%.3f, %.3f]\n",
        length(Vs), length(chain), minimum(Vs), maximum(Vs))
@printf("max ||F||inf over sweep = %.3e\n", maxF)

for V in (-0.300, -1.200, -2.500)
    k = argmin(abs.(Vs .- V)); u = Us[k]
    c = unpack_log_conc(u, m.cl.N)
    iCO, iH2 = compute_currents(u, Vs[k] - V_RHE_to_SHE_offset, m.cl)
    pH1 = -log10(c[1,1]/1000); pHN = -log10(c[1,end]/1000)
    @printf("V_RHE=%.3f: pH %.2f (x=0) → %.2f (x=L), c_CO2 %.3f → %.3f, i_CO=%.2f i_H2=%.2f mA/cm², FE_CO=%.1f%%\n",
            Vs[k], pH1, pHN, c[5,1], c[5,end], iCO/10, iH2/10,
            100*iCO/(iCO+iH2))
end
kpk = argmax([begin iCO,iH2 = compute_currents(Us[k], Vs[k]-V_RHE_to_SHE_offset, m.cl); iCO/(iCO+iH2) end for k in eachindex(Vs)])
iCOp, iH2p = compute_currents(Us[kpk], Vs[kpk]-V_RHE_to_SHE_offset, m.cl)
@printf("peak FE_CO = %.1f%% at V_RHE = %.3f\n", 100*iCOp/(iCOp+iH2p), Vs[kpk])
println("\nSTAGE 1 COMPLETE — generate plots and review before Stage 2.")
