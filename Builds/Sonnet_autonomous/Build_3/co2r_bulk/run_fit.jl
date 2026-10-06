# run_fit.jl — Fit step (after Stage 4): i0_CO2R, lambda_MHC, HER_prefactor,
# alpha_c_HER against the 5 experimental voltages, using the Stage 4
# residual (phi_L=0, phi_s=V_applied_SHE) for speed. §17.

include("params.jl")
include("mesh.jl")
include("chemistry.jl")
include("kinetics.jl")
include("transport.jl")
include("gas_transport.jl")
include("assembly.jl")
include("newton_affine.jl")
include("driver.jl")

using Printf
using Optim

const FIT_CACHE = joinpath(@__DIR__, "output_fit", "cache")
mkpath(FIT_CACHE)

function save_fit(u::Vector{Float64}, V::Float64)
    fn = joinpath(FIT_CACHE, @sprintf("fit_V%.6f.bin", V))
    open(fn, "w") do f
        write(f, Int64(length(u)))
        write(f, u)
    end
end

function main()
    @printf("=== Fit step (§17) ===\n")
    cl = make_cl_mesh(); gdl = make_gdl_mesh()
    ly = GasLayout(cl.N, gdl.N)
    Ntot = n_dofs_34(ly)
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq_vec = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    blocks = [1:5*cl.N, (5*cl.N+1):(5*cl.N+4*gdl.N), (5*cl.N+4*gdl.N+1):Ntot]

    # ── §17.3: load the Stage 4 grid cache for warm-starting the polish ────
    s4dir = joinpath(@__DIR__, "output_stage4", "cache")
    grid_cache = Dict{Float64,Vector{Float64}}()
    for V in V_RHE_grid
        fn = joinpath(s4dir, @sprintf("s4_V%.6f.bin", V))
        isfile(fn) || error("Stage 4 cache missing at V=$V — run run_stage4.jl first")
        u = open(fn, "r") do f
            n = read(f, Int64); read!(f, Vector{Float64}(undef, n))
        end
        grid_cache[V] = u
    end
    @printf("Loaded Stage 4 grid cache: %d voltages\n", length(grid_cache))

    plu = AffineLU()
    exp_cache = Dict{Float64,Vector{Float64}}()

    # ── DEVIATION (documented in BUILD_DIALOGUE.md): always warm-start each
    # polish from the fixed, known-good Stage-4 GRID cache rather than
    # chaining from the previous NM iteration's exp_cache entry (as the
    # guide's §17.3/§17.4 pseudocode literally does). A live debugging
    # session found that chaining lets one wild NM trial (e.g. a large
    # lambda_MHC jump that nearly kills CO2R at the mildest voltage) leave
    # a degenerate, near-zero-current state in exp_cache; every subsequent
    # eval — including the FINAL one after restoring the best-found
    # parameters — then warm-starts Newton from that bad state and
    # reconverges to a nearby but WRONG local solution instead of recovering
    # the correct branch, silently corrupting the reported fit. Re-anchoring
    # every polish to the grid cache costs a few extra Newton iterations
    # (~3 instead of ~2 per voltage, confirmed negligible — the grid cache
    # is always within one 0.025 V step of any experimental voltage) and
    # eliminates the corruption path entirely, since a fresh, physically
    # valid state is used on every single evaluation.
    function polish_all!()
        ok_all = true
        for V_rhe in E_exp_RHE
            idx_nearest = argmin(abs.(V_RHE_grid .- V_rhe))
            u_warm = copy(grid_cache[V_RHE_grid[idx_nearest]])
            res! = (F_, u_) -> residual_stage4!(F_, u_, V_rhe, cl, gdl, c_eq_vec, ly)
            info = newton_affine!(u_warm, res!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                                   max_iter=MAX_ITER_SWEEP, alpha_init=1.0, n_log_dofs=5*cl.N)
            if !info.converged
                info = newton_affine!(u_warm, res!; blocks=blocks, plu=plu, rtol=RTOL_NEWTON,
                                       max_iter=MAX_ITER_ANCHOR, alpha_init=ALPHA_INIT_ANCHOR, n_log_dofs=5*cl.N)
            end
            exp_cache[V_rhe] = u_warm
            ok_all &= info.converged
        end
        return ok_all
    end

    @printf("Initial polish at default kinetics (i0=%.1f, lambda=%.2f, HER_pf=%.3e, alpha_c=%.4f)...\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    ok0 = polish_all!()
    @assert ok0 "initial polish at default (pre-fit) kinetics failed to converge"
    @printf("Initial polish OK.\n")

    # ── §17.2 bounds; sloped penalty (v7.1 correction — NEVER a flat constant) ─
    lower = [10.0, 0.5, 8.0, 0.01]
    upper = [5000.0, 3.0, 18.0, 0.50]
    p_last = Ref([540.0, 1.8, log10(2.409e13), 0.0872])
    neval = Ref(0)

    function fitting_objective(params::Vector{Float64})
        neval[] += 1
        viol = 0.0
        for j in 1:4
            if params[j] < lower[j]
                viol += ((lower[j] - params[j]) / (upper[j] - lower[j]))^2
            elseif params[j] > upper[j]
                viol += ((params[j] - upper[j]) / (upper[j] - lower[j]))^2
            end
        end
        if viol > 0
            return 1e6 * (1 + viol)
        end
        set_i0_CO2R!(params[1]); set_lambda_MHC!(params[2])
        set_HER_prefactor!(10.0^params[3]); set_alpha_c_HER!(params[4])
        ok = polish_all!()
        if !ok
            return 1e6 * (1 + sum(((params .- p_last[]) ./ (upper .- lower)) .^ 2))
        end
        p_last[] = copy(params)
        sse = 0.0
        for (k, V_rhe) in enumerate(E_exp_RHE)
            u = exp_cache[V_rhe]
            V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
            iCO = compute_i_CO(u, V_applied_SHE, cl) / 10.0
            iH2 = compute_i_H2(u, V_applied_SHE, cl) / 10.0
            sse += (log10(max(iCO, 0.1)) - log10(i_CO_exp[k]))^2
            sse += (log10(max(iH2, 0.1)) - log10(i_H2_exp[k]))^2
        end
        return sse
    end

    params0 = [540.0, 1.8, log10(2.409e13), 0.0872]
    @printf("Starting Nelder-Mead (params0 = %s)...\n", params0)
    # DEVIATION (documented in BUILD_DIALOGUE.md): use a gentler initial
    # simplex than Optim.jl's NelderMead default (AffineSimplexer(a=0.025,
    # b=0.5), i.e. +150% per coordinate). At that default, perturbing
    # log10(HER_pf)=13.38 alone already overshoots the upper bound (18) on
    # the very first simplex vertex, and a live debugging run showed NM
    # making no progress at all in 200 iterations from this starting point.
    # A gentler a=0.01,b=0.1 (+10%) simplex stays inside the bounded,
    # sensitive parameter space and lets NM actually explore locally before
    # taking larger steps.
    simplexer = Optim.AffineSimplexer(a=0.01, b=0.1)
    result = optimize(fitting_objective, params0, NelderMead(initial_simplex=simplexer),
                       Optim.Options(iterations=250, f_reltol=1e-8, show_trace=true, show_every=5))
    println(result)

    pbest = Optim.minimizer(result)
    set_i0_CO2R!(pbest[1]); set_lambda_MHC!(pbest[2])
    set_HER_prefactor!(10.0^pbest[3]); set_alpha_c_HER!(pbest[4])
    @printf("\n=== Fitted kinetics ===\n")
    @printf("i0_CO2R      = %.3f A/m^2\n", i0_CO2R())
    @printf("lambda_MHC   = %.4f eV\n", lambda_MHC())
    @printf("HER_prefactor= %.4e\n", HER_prefactor())
    @printf("alpha_c_HER  = %.5f\n", alpha_c_HER())
    @printf("final SSE = %.5f, function evals = %d\n", Optim.minimum(result), neval[])

    # ── Per-point acceptance check (§17.6) ────────────────────────────────
    polish_all!()
    @printf("\nPer-point current comparison at the 5 experimental voltages:\n")
    all_pass = true
    for (k, V_rhe) in enumerate(E_exp_RHE)
        u = exp_cache[V_rhe]
        V_applied_SHE = V_rhe - V_RHE_to_SHE_offset
        iCO = compute_i_CO(u, V_applied_SHE, cl) / 10.0
        iH2 = compute_i_H2(u, V_applied_SHE, cl) / 10.0
        tol_CO = k == 5 ? 0.80 : 0.30
        errCO = abs(iCO - i_CO_exp[k]) / i_CO_exp[k]
        errH2 = abs(iH2 - i_H2_exp[k]) / i_H2_exp[k]
        passCO = errCO <= tol_CO; passH2 = errH2 <= 0.30
        all_pass &= (passCO && passH2)
        @printf("  V_RHE=%.3f: i_CO model=%.2f exp=%.2f (err=%.1f%%, tol=%.0f%%, %s) | i_H2 model=%.2f exp=%.2f (err=%.1f%%, %s)\n",
                V_rhe, iCO, i_CO_exp[k], 100*errCO, 100*tol_CO, passCO ? "PASS" : "FAIL",
                iH2, i_H2_exp[k], 100*errH2, passH2 ? "PASS" : "FAIL")
    end
    @printf("All per-point tolerances met: %s\n", all_pass)

    # ── §17.8: final 89-voltage Stage 4 sweep at fitted kinetics (diagnostic) ─
    @printf("\nRunning final 89-voltage Stage 4 sweep at fitted kinetics -> output_fit/cache/\n")
    u0 = make_u0_stage34(c_eq_vec, cl, gdl, ly)
    res_at_V! = (F_, u_, V_) -> residual_stage4!(F_, u_, V_, cl, gdl, c_eq_vec, ly)
    result_sweep = run_stage_sweep!(res_at_V!, V_RHE_grid, u0; blocks=blocks, n_log_dofs=5*cl.N,
                                     save_fn=save_fit, stage_name="Fit")
    n_ok = count(result_sweep.converged)
    @printf("Fit-kinetics Stage-4-physics sweep: %d / %d converged\n", n_ok, length(result_sweep.voltages))

    outdir = joinpath(@__DIR__, "output_fit")
    open(joinpath(outdir, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,i_CO_mA_cm2,i_H2_mA_cm2,i_total_mA_cm2,FE_CO,FE_H2,converged,iters,resid")
        for (k, V) in enumerate(result_sweep.voltages)
            u = result_sweep.solutions[k]
            V_SHE = V - V_RHE_to_SHE_offset
            iCO = compute_i_CO(u, V_SHE, cl) / 10.0
            iH2 = compute_i_H2(u, V_SHE, cl) / 10.0
            itot = iCO + iH2
            feCO = itot > 0 ? 100*iCO/itot : 0.0
            feH2 = itot > 0 ? 100*iH2/itot : 0.0
            println(f, "$V,$V_SHE,$iCO,$iH2,$itot,$feCO,$feH2,$(result_sweep.converged[k]),$(result_sweep.iters[k]),$(result_sweep.resid[k])")
        end
    end
    open(joinpath(outdir, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_Cs,c_CO2,R_CO2R,R_HER")
        for (k, V) in enumerate(result_sweep.voltages)
            u = result_sweep.solutions[k]
            V_SHE = V - V_RHE_to_SHE_offset
            c = unpack_log_conc(u, cl.N)
            for ix in 1:cl.N
                c_ix = c[:,ix]
                cCs = cCs_cell(c_ix)
                pH = -log10(max(c_ix[1]/1000.0, 1e-30))
                r_co2r = R_CO2R(c_ix[5], V_SHE, 0.0, c_ix[1], a_v_CL, lambda_MHC(), i0_CO2R(), n_e)
                r_her = R_HER(V_SHE, 0.0, c_ix[1], a_v_CL, n_e)
                x_um = cl.x_centers[ix]*1e6
                println(f, "$V,$x_um,$pH,$(c_ix[1]),$(c_ix[2]),$(c_ix[3]),$(c_ix[4]),$cCs,$(c_ix[5]),$r_co2r,$r_her")
            end
        end
    end
    open(joinpath(outdir, "gas_profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,p_CO2_atm,p_CO_atm,p_H2_atm,p_total_Pa")
        for (k, V) in enumerate(result_sweep.voltages)
            u = result_sweep.solutions[k]
            for ig in 1:gdl.N
                om = gdl_omega(u, ly, ig); p = gdl_p(u, ly, ig)
                Mn = Mn_from_omega(om); x = omega_to_x(om, Mn)
                x_um = gdl.x_centers[ig]*1e6
                println(f, "$V,$x_um,$(x[1]*p/101325),$(x[2]*p/101325),$(x[3]*p/101325),$p")
            end
            for ix in 1:cl.N
                om = cl_omega(u, ly, ix); p = cl_p(u, ly, ix)
                Mn = Mn_from_omega(om); x = omega_to_x(om, Mn)
                x_um = cl.x_centers[ix]*1e6
                println(f, "$V,$x_um,$(x[1]*p/101325),$(x[2]*p/101325),$(x[3]*p/101325),$p")
            end
        end
    end
    @printf("Wrote polarization.csv, profiles.csv, gas_profiles.csv\n")

    open(joinpath(outdir, "fitted_params.txt"), "w") do f
        @printf(f, "i0_CO2R=%.6f\n", i0_CO2R())
        @printf(f, "lambda_MHC=%.6f\n", lambda_MHC())
        @printf(f, "HER_prefactor=%.6e\n", HER_prefactor())
        @printf(f, "alpha_c_HER=%.6f\n", alpha_c_HER())
    end
    @printf("Wrote fitted_params.txt\n")

    return result, result_sweep, cl, gdl, ly
end

fit_result, fit_sweep, cl_fit, gdl_fit, ly_fit = main()
