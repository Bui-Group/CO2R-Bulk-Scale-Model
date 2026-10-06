# run_fit.jl — Kinetic parameter fit (§17), run AFTER Stage 4.
# Fits (i0_CO2R, λ_MHC, log10 HER_prefactor, α_c) to the 5 Lees voltages using the
# Stage-4 residual (simplified kinetics φ_L=0, φ_s=V_applied) for speed. Polishes at
# the exact experimental voltages each NM evaluation with the §9.5 kernel (per-voltage
# `newton_affine!`, warm-started from the running exp_cache / the Stage-4 grid cache —
# the §9.5.5 "Fit" row), then writes a post-fit 89-V Stage-4 refit sweep to
# output_fit/cache/fit_V*.bin — the Stage-4-physics reference at fitted kinetics that
# the Stage 5 checks compare against (§17.8). Writes output_fit/fitted_params.jl.

const HERE = @__DIR__
for f in ["params.jl","mesh.jl","chemistry.jl","kinetics.jl","transport.jl",
          "gas_transport.jl","solver.jl","cache.jl",
          "assembly.jl","assembly_stage3.jl","assembly_stage4.jl","plots.jl"]
    include(joinpath(HERE, f))
end
using Optim

const STAGE4_CACHE = joinpath(HERE, "output_stage4", "cache")
const FIT_CACHE    = joinpath(HERE, "output_fit", "cache")
const N_CL  = parse(Int, get(ENV, "MESH_N_CL", "80"))
const N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))

const cl_mesh  = make_cl_mesh(N_CL)
const gdl_mesh = make_gdl_mesh(N_GDL)
const _eq      = solve_buffer_equilibrium(1000.0, 1000.0)
const c_eq     = [_eq.H, _eq.OH, _eq.HCO3, _eq.CO3, _eq.CO2]
const ND       = n_dof_stage34(N_CL, N_GDL)
const nlog     = 5 * N_CL

# §9.5 workspace shared by every polish. The kinetics are Ref-backed, so one residual
# closure / one AD assembler serves the whole fit (no rebuild on parameter change).
const V_ref = Ref(0.0)
res_closure!(F, uu) = residual_stage4!(F, uu, V_ref[], cl_mesh, gdl_mesh, c_eq)
const Jtemplate = let (r, c) = build_gas_pattern(N_CL, N_GDL); pattern_matrix(r, c, ND) end
const plu       = AffineLU()
const blocks    = dof_blocks(4, N_CL, N_GDL)
const assemble! = make_ad_assembler(res_closure!, ND; chunk=12)

const s4_grid_V = list_cache(STAGE4_CACHE, "s4_")
nearest_s4(V) = load_solution(STAGE4_CACHE, "s4_", s4_grid_V[argmin(abs.(s4_grid_V .- V))])

const exp_cache = Dict{Float64, Vector{Float64}}()

# NM parameter bounds (§17.2)
const lower = [10.0, 0.5, 8.0, 0.01]
const upper = [5000.0, 3.0, 18.0, 0.50]

"Per-voltage polish (§17.3, §9.5.5 Fit row): same kernel, no continuation. Warm start from the
running exp_cache (else the nearest Stage-4 grid solution); on failure retry from the Stage-4
grid neighbour with cautious damping (alpha_init = 1e-2)."
function polish_state!(u::Vector{Float64}, V_rhe::Float64; fallback::Union{Nothing,Vector{Float64}}=nothing)
    V_ref[] = V_rhe
    info = newton_affine!(u, res_closure!; blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate,
                          alpha_init=1.0, max_iter=40, n_log_dofs=nlog)
    if !info.converged && fallback !== nothing
        u .= fallback
        info = newton_affine!(u, res_closure!; blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate,
                              alpha_init=1e-2, max_iter=200, n_log_dofs=nlog)
    end
    return info
end

"Solve Stage 4 at the exact experimental voltage V_rhe with the current kinetics."
function polish!(V_rhe::Float64)
    u = haskey(exp_cache, V_rhe) ? copy(exp_cache[V_rhe]) : copy(nearest_s4(V_rhe))
    info = polish_state!(u, V_rhe; fallback=nearest_s4(V_rhe))
    info.converged && (exp_cache[V_rhe] = u)
    return info.converged
end

function model_currents(V_rhe::Float64)
    u = exp_cache[V_rhe]; Vsh = V_rhe - V_RHE_to_SHE_offset
    return compute_i_CO(u, Vsh, cl_mesh)/10, compute_i_H2(u, Vsh, cl_mesh)/10   # mA/cm²
end

function fitting_objective(p::Vector{Float64})
    (any(p .< lower) || any(p .> upper)) && return 1e6
    set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]); set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4])
    sse = 0.0
    for (k, V) in enumerate(E_exp_RHE)
        polish!(V) || return 1e6
        iCO, iH2 = model_currents(V)
        sse += (log10(max(iCO, 0.1)) - log10(i_CO_exp[k]))^2
        sse += (log10(max(iH2, 0.1)) - log10(i_H2_exp[k]))^2
    end
    return sse
end

function main()
    println("="^70)
    println("=== Fit step (§17): Nelder-Mead on (i0_CO2R, λ_MHC, log10 HER_pf, α_c) ===")
    @printf("Start: i0=%.0f λ=%.2f HER_pf=%.3e α_c=%.4f\n", i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    println("Solver: §9.5 kernel newton_affine! (per-voltage polish, warm-started from the Stage-4 cache)")
    println("="^70)
    isempty(s4_grid_V) && error("No Stage 4 cache — run run_stage4.jl first.")

    # initial polish at the experimental voltages (default kinetics)
    for V in E_exp_RHE
        polish!(V) || error("initial polish failed at V_RHE=$V")
    end
    println("Initial polish complete at the 5 experimental voltages.")

    params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
    @printf("Initial SSE = %.4f\n", fitting_objective(params0))

    t0 = time()
    result = optimize(fitting_objective, params0, NelderMead(),
                      Optim.Options(iterations=200, g_tol=1e-6, show_trace=false))
    pf = Optim.minimizer(result)
    set_i0_CO2R!(pf[1]); set_lambda_MHC!(pf[2]); set_HER_prefactor!(10.0^pf[3]); set_alpha_c_HER!(pf[4])

    println("\n--- FITTED PARAMETERS ---")
    @printf("i0_CO2R       = %.2f A/m²\n", i0_CO2R())
    @printf("λ_MHC         = %.4f eV\n", lambda_MHC())
    @printf("HER_prefactor = %.4e mA/cm²\n", HER_prefactor())
    @printf("α_c_HER       = %.5f\n", alpha_c_HER())
    @printf("final SSE     = %.5f  (%d NM iters, %d f-evals, %.1f s)\n",
            Optim.minimum(result), Optim.iterations(result), Optim.f_calls(result), time()-t0)

    # persist the fitted kinetics for run_stage5.jl (§17.8, gotcha #14)
    pdir = joinpath(HERE, "output_fit"); mkpath(pdir)
    open(joinpath(pdir, "fitted_params.jl"), "w") do f
        println(f, "# Fitted kinetic parameters from run_fit.jl (§17). Loaded by run_stage5.jl.")
        @printf(f, "set_i0_CO2R!(%.2f)\nset_lambda_MHC!(%.4f)\nset_HER_prefactor!(%.4e)\nset_alpha_c_HER!(%.5f)\n",
                i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    end
    println("Wrote output_fit/fitted_params.jl")

    # Per-point fit quality (§17.6)
    println("\nFit quality vs Lees:")
    for (k, V) in enumerate(E_exp_RHE)
        polish!(V); iCO, iH2 = model_currents(V)
        @printf("  V_RHE=%.3f  i_CO %.1f vs %.1f (%.0f%%)  i_H2 %.1f vs %.1f (%.0f%%)  FE_CO %.1f%% vs %.1f%%\n",
                V, iCO, i_CO_exp[k], 100*abs(iCO-i_CO_exp[k])/i_CO_exp[k],
                iH2, i_H2_exp[k], 100*abs(iH2-i_H2_exp[k])/i_H2_exp[k],
                100*iCO/(iCO+iH2), FE_CO_exp[k])
    end

    # §17.8 post-fit 89-V Stage-4 refit sweep → output_fit/cache (the Stage-4-physics reference at
    # fitted kinetics for the Stage 5 checks; per-voltage polish, carried forward, Stage-4 grid fallback)
    println("\nWriting post-fit 89-V Stage-4 refit sweep to output_fit/cache ...")
    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    u = copy(nearest_s4(Vs[1]))
    fit_results = Tuple{Float64, Vector{Float64}}[]; nconv = 0; maxF = 0.0
    for V in Vs
        info = polish_state!(u, V; fallback=nearest_s4(V))
        if info.converged
            save_solution(FIT_CACHE, "fit_", copy(u), V); push!(fit_results, (V, copy(u))); nconv += 1
            maxF = max(maxF, info.residual)
        else
            @printf("  [refit FAIL] V_RHE=%.4f reason=%s ‖F‖∞=%.2e\n", V, info.reason, info.residual)
            u = copy(nearest_s4(V))
        end
    end
    sort!(fit_results, by = r -> r[1])
    @printf("Refit sweep: %d/%d converged (max ‖F‖∞ = %.2e) → output_fit/cache/fit_V*.bin\n", nconv, length(Vs), maxF)
    if !isempty(fit_results)
        iCOs = [compute_i_CO(r[2], r[1]-V_RHE_to_SHE_offset, cl_mesh)/10 for r in fit_results]
        iH2s = [compute_i_H2(r[2], r[1]-V_RHE_to_SHE_offset, cl_mesh)/10 for r in fit_results]
        FE = 100 .* iCOs ./ max.(iCOs .+ iH2s, 1e-12)
        @printf("  refit sweep: i_CO peak %.1f mA/cm² at V_RHE=%.3f; FE_CO peak %.1f%% at V_RHE=%.3f\n",
                maximum(iCOs), fit_results[argmax(iCOs)][1], maximum(FE), fit_results[argmax(FE)][1])
    end

    # Plots (Fit step: A1/A2/B1 with fitted kinetics)
    plot_A1(fit_results, cl_mesh, joinpath(pdir, "fit_A1_polarization.png"); stage_label="Fit (Stage 4 + fitted kinetics)")
    plot_A2(fit_results, cl_mesh, joinpath(pdir, "fit_A2_liquid.png"); stage_label="Fit")
    plot_B1(fit_results, cl_mesh, gdl_mesh, joinpath(pdir, "fit_B1_gas.png"); stage_label="Fit")
    s4 = [(V, load_solution(STAGE4_CACHE,"s4_",V)) for V in s4_grid_V]
    plot_polarization_overlay([s4, fit_results], ["S4 prefit","Fit"], cl_mesh,
        joinpath(pdir, "fit_overlay_vs_stage4.png"); title="Stage 4 pre-fit vs fitted kinetics")
    println("Plots: fit_A1_polarization.png, fit_A2_liquid.png, fit_B1_gas.png, fit_overlay_vs_stage4.png")

    println("\n" * "="^70)
    println("FIT STEP COMPLETE — STOP GATE. Review fitted params + plots before Stage 5.")
    println("="^70)
end

main()
