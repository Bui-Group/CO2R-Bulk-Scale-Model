# run_fit.jl — Fit step after Stage 4 (§17). Nelder-Mead on 4 kinetic params
# (i0_CO2R, λ_MHC, log10 HER_prefactor, α_c) vs the 5 Lees points, using the Stage 4 residual
# (φ_L=0, φ_s=V_applied). Polishes at the EXACT experimental voltages with the §9.5.3 kernel
# (per-voltage `newton_affine!`, warm-started from the Stage 4 cache — §9.5.5 Fit row, §17.3).
# After the fit, a final 89-V Stage 4 sweep at the fitted kinetics is written to
# output_fit/cache/fit_V*.bin — the Stage-4-physics reference at fitted kinetics that the Stage 5
# checks compare against (§17.8; Stage 5 itself is seeded from the V_c = 0 anchor, not from it).
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "plots.jl"))
using Printf, Optim

const STAGE4_CACHE = joinpath(HERE, "output_stage4", "cache")
const FIT_CACHE    = joinpath(HERE, "output_fit", "cache")
const FIT_PLOTS    = joinpath(HERE, "output_fit")

function main()
    println("="^70)
    println("=== Fit step (after Stage 4): Nelder-Mead on 4 kinetic parameters ===")
    @printf("Start: i0_CO2R=%.0f λ=%.2f HER_pf=%.3e α_c=%.4f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    println("="^70)
    isempty(list_cache(STAGE4_CACHE, "s4")) && error("No Stage 4 cache — run run_stage4.jl first")

    cl_mesh = make_cl_mesh(); gdl_mesh = make_gdl_mesh(); gas_mesh = GasMesh(gdl_mesh, cl_mesh)
    c_eq_vec = equilibrium_c_vec(1000.0, 1000.0)
    N = 5*N_CL + 4*(N_GDL+N_CL)

    residual_at_V!(F, u, V) = residual_stage4!(F, u, V, cl_mesh, gdl_mesh, gas_mesh, c_eq_vec)
    s4V = list_cache(STAGE4_CACHE, "s4")
    grid_seed(V) = copy(load_solution(STAGE4_CACHE, "s4", s4V[argmin(abs.(s4V .- V))]))

    Jpat = probe_sparsity((F, u) -> residual_at_V!(F, u, -1.0), N; u_seed=grid_seed(-1.0))
    plu = AffineLU()
    blocks = dof_blocks(4)
    V_ref = Ref(0.0)
    res_closure!(F, u) = residual_at_V!(F, u, V_ref[])
    assemble! = make_ad_assembler(res_closure!, N; chunk=12)   # Ref-backed kinetics ⇒ no rebuild on param change

    # Per-voltage polish (§9.5.5 Fit row, §17.3): same kernel, no continuation.
    # `aggressive` = cautious start (alpha_init=1e-2) with a generous cap, for retries from the grid seed.
    function newton_polish!(u::Vector{Float64}, V::Float64; aggressive::Bool=false)
        V_ref[] = V
        newton_affine!(u, res_closure!; blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jpat,
                       alpha_init = aggressive ? 1e-2 : 1.0,
                       max_iter   = aggressive ? 200  : 40, n_log_dofs=5*N_CL)
    end

    # Polish at the exact experimental voltages (§17.3). Warm-start from exp_cache (or grid).
    exp_cache = Dict{Float64, Vector{Float64}}()
    function polish!(from_grid::Bool)
        for V in E_exp_RHE
            u = (from_grid || !haskey(exp_cache, V)) ? grid_seed(V) : copy(exp_cache[V])
            info = newton_polish!(u, V)
            if !info.converged && from_grid          # initial polish: retry from the grid seed with cautious damping
                u = grid_seed(V); info = newton_polish!(u, V; aggressive=true)
            end
            info.converged || return false
            exp_cache[V] = u
        end
        return true
    end

    # Fitting parameters + bounds (§17.2)
    params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
    lower   = [ 10.0, 0.5,  8.0, 0.01]
    upper   = [5000.0, 3.0, 18.0, 0.50]

    neval = Ref(0)
    function objective(p)
        for i in 1:4
            (p[i] < lower[i] || p[i] > upper[i]) && return 1e6
        end
        set_i0_CO2R!(p[1]); set_lambda_MHC!(p[2]); set_HER_prefactor!(10.0^p[3]); set_alpha_c_HER!(p[4])
        polish!(false) || return 1e6
        sse = 0.0
        for (k, V) in enumerate(E_exp_RHE)
            Va = V - V_RHE_to_SHE_offset
            iCO = compute_i_CO(exp_cache[V], Va, cl_mesh)/10
            iH2 = compute_i_H2(exp_cache[V], Va, cl_mesh)/10
            sse += (log10(max(iCO,0.1)) - log10(i_CO_exp[k]))^2
            sse += (log10(max(iH2,0.1)) - log10(i_H2_exp[k]))^2
        end
        neval[] += 1
        neval[] % 10 == 0 && @printf("  eval %3d: i0=%.1f λ=%.3f HER_pf=%.3e α_c=%.4f  sse=%.5f\n",
                                     neval[], p[1], p[2], 10.0^p[3], p[4], sse)
        return sse
    end

    # Initial polish from grid, then Nelder-Mead
    println("Initial polish at experimental voltages (default kinetics)...")
    @assert polish!(true) "initial polish failed"
    t0 = time()
    result = optimize(objective, params0, NelderMead(), Optim.Options(iterations=200, show_trace=false))
    pfit = Optim.minimizer(result)
    @printf("\nNelder-Mead done in %.1fs, %d evals, final SSE=%.5f\n",
            time()-t0, neval[], Optim.minimum(result))

    set_i0_CO2R!(pfit[1]); set_lambda_MHC!(pfit[2]); set_HER_prefactor!(10.0^pfit[3]); set_alpha_c_HER!(pfit[4])
    @printf("\nFITTED: i0_CO2R=%.2f A/m², λ_MHC=%.4f eV, HER_pf=%.4e, α_c=%.5f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())

    # Save fitted params for Stage 5
    mkpath(FIT_PLOTS)
    open(joinpath(FIT_PLOTS, "fitted_params.txt"), "w") do io
        @printf(io, "%.10e\n%.10e\n%.10e\n%.10e\n", i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    end

    # ── Report per-point currents + acceptance criteria (§17.6) ──
    println("\n── Fit quality vs Lees et al. (§17.6) ──")
    @printf("%-8s %10s %10s %8s | %10s %10s %8s\n","V_RHE","iCO_mod","iCO_exp","err%","iH2_mod","iH2_exp","err%")
    polish!(false)
    fit_iCO = Float64[]; fit_iH2 = Float64[]
    for (k, V) in enumerate(E_exp_RHE)
        Va = V - V_RHE_to_SHE_offset
        iCO = compute_i_CO(exp_cache[V], Va, cl_mesh)/10; iH2 = compute_i_H2(exp_cache[V], Va, cl_mesh)/10
        push!(fit_iCO, iCO); push!(fit_iH2, iH2)
        eCO = 100*abs(iCO-i_CO_exp[k])/i_CO_exp[k]; eH2 = 100*abs(iH2-i_H2_exp[k])/i_H2_exp[k]
        @printf("%-8.3f %10.1f %10.1f %7.1f | %10.1f %10.1f %7.1f\n",
                V, iCO, i_CO_exp[k], eCO, iH2, i_H2_exp[k], eH2)
    end
    FE_CO_fit = fit_iCO ./ (fit_iCO .+ fit_iH2) .* 100
    @printf("FE_CO model at 5 pts: %s (exp: %s)\n",
            join([@sprintf("%.0f",x) for x in FE_CO_fit],","), join([@sprintf("%.0f",x) for x in FE_CO_exp],","))

    # ── Final 89-V Stage 4 sweep at fitted kinetics → output_fit/cache/fit_V*.bin (§17.8) ──
    # Per-voltage polish with the fitted kinetics, warm-started from the previous refit state; on a
    # failure, re-seed from the Stage 4 grid cache at that voltage and retry with cautious damping.
    println("\nFinal 89-V Stage 4 sweep at fitted kinetics → output_fit/cache/ ...")
    save_fit(u, V) = save_solution(FIT_CACHE, "fit", u, V)
    cur(u, V) = (compute_i_CO(u, V-V_RHE_to_SHE_offset, cl_mesh)/10, compute_i_H2(u, V-V_RHE_to_SHE_offset, cl_mesh)/10)
    Vs = sort(collect(V_RHE_grid), by = v -> -v)          # mild → cathodic
    results = Any[]; n_conv = 0; n_tot = length(Vs); maxres = 0.0
    t1 = time()
    u = grid_seed(Vs[1])
    for V in Vs
        info = newton_polish!(u, V)
        if !info.converged
            u = grid_seed(V); info = newton_polish!(u, V; aggressive=true)
        end
        if info.converged
            save_fit(u, V)
            n_conv += 1; maxres = max(maxres, info.residual)
            c2 = cur(u, V); push!(results, (V, c2...))
            @printf("Fit-sweep V_RHE=%.3f  ‖F‖∞=%.2e iter=%2d  %.2f %.2f\n", V, info.residual, info.iterations, c2...)
        else
            @printf("Fit-sweep V_RHE=%.3f  FAILED (%s, ‖F‖∞=%.2e)\n", V, info.reason, info.residual)
            u = grid_seed(V)                              # avoid carrying a diverged state forward
        end
    end
    @printf("Fit-sweep: %d/%d converged in %.1fs, max‖F‖∞=%.2e\n", n_conv, n_tot, time()-t1, maxres)
    sort!(results, by = r -> r[1])

    # ── Plots: fitted polarization + FE, overlaid with experiment ──
    Vr = [r[1] for r in results]; iCO = [r[2] for r in results]; iH2 = [r[3] for r in results]
    plot_A1(FIT_PLOTS, Vr, iCO, iH2; title_prefix="Fit (Stage 4, fitted kinetics)")
    # A2/B1 at fitted kinetics
    Vcache = sort(list_cache(FIT_CACHE, "fit"), by = v -> -v)
    profiles = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64}}[]
    gas_series = Tuple{Float64,Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    for V in Vcache
        uu = load_solution(FIT_CACHE, "fit", V); c = unpack_log_conc(uu, N_CL); Va = V - V_RHE_to_SHE_offset
        rco2r = [R_CO2R(c[5,ix], Va, 0.0, c[1,ix], a_v_CL, lambda_MHC(), i0_CO2R(), n_e) for ix in 1:N_CL]
        rher  = [R_HER(Va, 0.0, c[1,ix], a_v_CL, n_e) for ix in 1:N_CL]
        push!(profiles, (V, c, rco2r, rher))
        pCO2,pCO,pH2,ptot = gas_profiles(uu, N_CL, N_GDL); push!(gas_series, (V,pCO2,pCO,pH2,ptot))
    end
    plot_A2(FIT_PLOTS, cl_mesh, profiles; title_prefix="Fit")
    plot_B1(FIT_PLOTS, vcat(gdl_mesh.xc, cl_mesh.xc).*1e6, gas_series; title_prefix="Fit")

    @printf("\nFit sweep: %d/%d converged, max‖F‖∞=%.2e\n", n_conv, n_tot, maxres)
    println(">>> FIT STEP COMPLETE. Fitted params in output_fit/fitted_params.txt. STOP for review. <<<")
    return pfit
end

main()
