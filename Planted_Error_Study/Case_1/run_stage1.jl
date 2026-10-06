# run_stage1.jl — Stage 1 driver (§Stage 1)
# Liquid NP diffusion only + Dirichlet BCs + simplified kinetics (φ_s=V_applied, φ_L=0).
# Anchors at the mildest grid voltage (−0.300 V_RHE) from the exact equilibrium guess,
# walks the 89-V grid by tangent-predictor continuation (§9.5.5), caches every
# converged solution, generates A1/A2 plots, STOPS.
#
# Mesh: production N_CL=80 by default; override with env MESH_N_CL (dev=20).

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "solver.jl"))
include(joinpath(HERE, "cache.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "plots.jl"))

const STAGE1_CACHE = joinpath(HERE, "output_stage1", "cache")
const N_CL  = parse(Int, get(ENV, "MESH_N_CL", "80"))
const N_GDL = parse(Int, get(ENV, "MESH_N_GDL", "120"))

"Number of derivative sign changes along a sampled curve (smoothness check)."
deriv_sign_changes(x) = (d = diff(x); count(k -> d[k]*d[k+1] < 0, 1:length(d)-1))

function main()
    println("="^70)
    println("=== Stage 1: Liquid NP (diffusion only) + Dirichlet BCs ===")
    @printf("i0_CO2R = %.0f A/m²,  λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf  = %.3e mA/cm²,  α_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL = %d  (DOFs = %d)\n", N_CL, 5*N_CL)
    @printf("Cache dir: %s\n", STAGE1_CACHE)
    println("="^70)

    cl_mesh = make_cl_mesh(N_CL)

    # ── Exact bulk equilibrium (§9.6) + verification ──
    eq = solve_buffer_equilibrium(1000.0, 1000.0)
    c_eq = [eq.H, eq.OH, eq.HCO3, eq.CO3, eq.CO2]
    @printf("Equilibrium: pH=%.3f  c_CO2=%.3f  c_HCO3=%.1f  c_CO3=%.2f  c_OH=%.3e\n",
            eq.pH, eq.CO2, eq.HCO3, eq.CO3, eq.OH)
    Sbuf = buffer_sources(c_eq)
    @printf("max|S_buf(c_eq)| = %.3e mol/m³/s  (must be < 1e-6)\n", maximum(abs, Sbuf))
    @assert maximum(abs, Sbuf) < 1e-6 "Buffer equilibrium not satisfied — check K/k unit conversions"
    cb = c_eq[1] + 1000.0 - c_eq[2] - c_eq[3] - 2*c_eq[4]
    @printf("charge balance residual = %.3e mol/m³\n", cb)

    # ── §9.5.5 continuation: anchor at −0.300 V_RHE from the equilibrium guess, then the
    #    tangent predictor down the grid (mild → cathodic) ──
    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    ln_c_eq = log.(c_eq)
    u = repeat(ln_c_eq, N_CL)                     # cell-major [lnH, lnOH, lnHCO3, lnCO3, lnCO2] × N_CL
    ND = 5 * N_CL
    rows, cols = block_tridiag_pattern(N_CL, 5)   # §9.3.3 fixed pattern (5×5 blocks, tridiagonal)
    Jtemplate = pattern_matrix(rows, cols, ND)
    plu = AffineLU()                              # §9.5.2 persistent sparse LU (stdlib)
    blocks = dof_blocks(1, N_CL)                  # §9.5.3 weight blocks
    V_ref = Ref(Vs[1])
    res_closure!(F, uu) = residual_stage1!(F, uu, V_ref[], cl_mesh, c_eq)
    assemble! = make_ad_assembler(res_closure!, ND; chunk=12)   # §9.3.2 AD Jacobian → fixed pattern
    println("Solver: §9.5 stack — newton_affine! + AffineLU (stdlib sparse LU) + ForwardDiff AD; §9.5.5 continuation")

    results = Tuple{Float64, Vector{Float64}}[]
    nconv = Ref(0); maxF = Ref(0.0)
    println("\nSweeping $(length(Vs)) voltages...")
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks=blocks, plu=plu, jac_assembler=assemble!, J=Jtemplate, n_log_dofs=ND,
        on_converged = (V, uu, info) -> begin
            save_solution(STAGE1_CACHE, "s1_", copy(uu), V)          # cache FIRST (§10.4)
            push!(results, (V, copy(uu))); nconv[] += 1
            maxF[] = max(maxF[], info.residual)
        end)
    t_sweep = time() - t0
    n_iter = sum(l.iterations for l in sweep_log)
    @printf("\nConverged: %d/%d voltages in %.1f s\n", nconv[], length(Vs), t_sweep)
    @printf("  anchor V_RHE=%.3f: %d iters, ||F||inf=%.2e | %d Newton iterations total, %.2f/step | max ||F||inf over sweep = %.2e (§9.5.6 level 1e-6)\n",
            Vs[1], sweep_log[1].iterations, sweep_log[1].residual, n_iter, n_iter/length(Vs), maxF[])

    sort!(results, by = r -> r[1])   # ascending V_RHE for plotting

    # ── Diagnostics summary ──
    if !isempty(results)
        Vmid = results[argmin(abs.([r[1] for r in results] .- (-1.205)))]
        Vsh = Vmid[1] - V_RHE_to_SHE_offset
        co = compute_i_CO(Vmid[2], Vsh, cl_mesh)/10
        h2 = compute_i_H2(Vmid[2], Vsh, cl_mesh)/10
        cM = unpack_log_conc(Vmid[2], N_CL)
        pH_el = -log10(cM[1,end]/1000.0)
        @printf("\nAt V_RHE≈%.3f: i_CO=%.1f i_H2=%.1f mA/cm²  FE_CO=%.1f%%  pH(EL face)=%.2f  c_CO2(EL)=%.3f\n",
                Vmid[1], co, h2, 100*co/(co+h2), pH_el, cM[5,end])

        # sweep-wide self-verification (the numbers cited in BUILD_DIALOGUE.md)
        Vr   = [r[1] for r in results]
        iCOs = [compute_i_CO(r[2], r[1]-V_RHE_to_SHE_offset, cl_mesh)/10 for r in results]
        iH2s = [compute_i_H2(r[2], r[1]-V_RHE_to_SHE_offset, cl_mesh)/10 for r in results]
        itot = iCOs .+ iH2s
        kpk  = argmax(iCOs)
        pHel = [-log10(unpack_log_conc(r[2], N_CL)[1,end]/1000.0) for r in results]
        eln  = 0.0
        for (V, uu) in results
            c = unpack_log_conc(uu, N_CL)
            for ix in 1:N_CL
                cCs = c[2,ix] + c[3,ix] + 2*c[4,ix] - c[1,ix] - C_fix*eps_I
                eln = max(eln, abs(c[1,ix] + cCs + C_fix*eps_I - c[2,ix] - c[3,ix] - 2*c[4,ix]))
            end
        end
        cmost = unpack_log_conc(results[1][2], N_CL)            # results[1] = most cathodic V
        ixm = argmin(cmost[5, :])
        println("\nSweep-wide self-verification (mild → cathodic):")
        @printf("  i_CO peak = %.1f mA/cm² at V_RHE = %.3f\n", iCOs[kpk], Vr[kpk])
        @printf("  i_total %.1f → %.0f mA/cm²; derivative sign changes: i_total %d, i_CO %d, i_H2 %d\n",
                itot[end], itot[1], deriv_sign_changes(itot), deriv_sign_changes(iCOs), deriv_sign_changes(iH2s))
        @printf("  pH(EL face) %.2f → %.2f\n", pHel[end], pHel[1])
        @printf("  max electroneutrality residual (Cs⁺ from e-neutrality) = %.1e mol/m³\n", eln)
        @printf("  c_CO2(aq) minimum at V_RHE=%.3f: %.2e mol/m³ at x=%.2f μm (cell %d of %d)\n",
                results[1][1], cmost[5, ixm], cl_mesh.x_centers[ixm]*1e6, ixm, N_CL)
        FEpk = 100*iCOs ./ max.(itot, 1e-12)
        @printf("  peak FE_CO = %.1f%% at V_RHE = %.3f\n", maximum(FEpk), Vr[argmax(FEpk)])
    end

    # ── Plots ──
    pdir = joinpath(HERE, "output_stage1")
    mkpath(pdir)
    plot_A1(results, cl_mesh, joinpath(pdir, "stage1_A1_polarization.png"); stage_label="Stage 1")
    plot_A2(results, cl_mesh, joinpath(pdir, "stage1_A2_liquid.png"); stage_label="Stage 1")
    println("Plots written: stage1_A1_polarization.png, stage1_A2_liquid.png")

    println("\n" * "="^70)
    println("STAGE 1 COMPLETE — STOP GATE. Review plots before Stage 2.")
    println("="^70)
    return nconv[], length(Vs)
end

main()
