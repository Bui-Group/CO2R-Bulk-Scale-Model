# run_stage1.jl — Stage 1: liquid NP diffusion only + Dirichlet both sides +
# simplified kinetics (φ_s = V_applied_SHE, φ_L = 0).  §11 Stage 1.
#
# Anchors at the mildest grid voltage (−0.300 V_RHE) from the exact equilibrium
# guess and walks the 89-voltage grid by tangent-predictor continuation (§9.5.5),
# caching each converged solution, dumping diagnostics CSVs, and generating the
# A1 / A2 plots. STOP gate for human review.

include(joinpath(@__DIR__, "co2r.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))

const OUT_DIR   = joinpath(HERE, "output_stage1")
const CACHE_DIR = joinpath(OUT_DIR, "cache")
mkpath(CACHE_DIR)

function main()
    @printf("=== Stage 1 ===  (N_CL=%d, N_GDL=%d)\n", N_CL, N_GDL)
    @printf("i0_CO2R = %.1f A/m², λ_MHC = %.2f eV, HER_pf = %.3e, α_c = %.4f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    @printf("Cache dir: %s\n", CACHE_DIR)

    clm = make_cl_mesh()
    c_eq = equilibrium_vec(1000.0, 1000.0)
    c_CO2_bc = H_eff_vantHoff() * (x_inlet[1] * p_total / 101325.0)   # van't Hoff only (Stage 1)
    @printf("c_CO2 Dirichlet @ GDL face = %.3f mol/m³;  c_eq = [%.3e, %.3e, %.2f, %.2f, %.2f]\n",
            c_CO2_bc, c_eq...)

    Ndof = 5 * clm.N

    # ── §9.5 solver workspace: fixed block-tridiagonal pattern (§9.3.3), persistent LU
    #    (§9.5.2), iterate-derived weight blocks (§9.5.3)
    prow, pcol = block_tridiag_pattern(clm.N, 5)
    Jt = pattern_matrix(prow, pcol, Ndof)
    plu = AffineLU()
    blocks = dof_blocks(1, clm.N, N_GDL)

    # equilibrium anchor guess (§9.5.5 table)
    u_eq = zeros(Ndof)
    for ix in 1:clm.N
        u_eq[5*(ix-1)+1:5*ix] .= log.(c_eq)
    end
    u = copy(u_eq)

    Vs = sort(collect(V_RHE_grid), by = v -> -v)   # mild (−0.3) → cathodic (−2.5)
    rows = Vector{NTuple{7,Float64}}()
    ok_Vs = Float64[]; sols = Dict{Float64,Vector{Float64}}()

    # the residual closure reads the voltage through V_ref, so a single AD config /
    # assembler (stable ForwardDiff tag) serves the whole sweep
    V_ref = Ref(Vs[1])
    res_closure!(F, u_) = residual_stage1!(F, u_, V_ref[], clm, c_CO2_bc, c_eq)
    assemble! = make_ad_assembler(res_closure!, Ndof)

    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jt, n_log_dofs = Ndof,
        on_converged = (V, uu, info) -> begin
            save_solution(CACHE_DIR, "s1_", copy(uu), V)
            push!(rows, polarization_row(copy(uu), V, clm, V - V_RHE_to_SHE_offset))
            push!(ok_Vs, V); sols[V] = copy(uu)
            @printf("  V_RHE=%.3f  ✓ %2d it  ‖F‖∞=%.2e  i_CO=%.1f i_H2=%.1f mA/cm²\n",
                    V, info.iterations, info.residual, rows[end][3], rows[end][4])
        end)
    t_sweep = time() - t0
    nfail  = count(l -> !l.converged, sweep_log)
    n_iter = sum(l.iterations for l in sweep_log)
    maxF   = maximum(l.residual for l in sweep_log if l.converged)
    @printf("\nSweep done: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step). max ‖F‖∞ = %.2e\n",
            length(ok_Vs), length(Vs), t_sweep, n_iter, n_iter/length(Vs), maxF)

    # ── diagnostics dump ──
    write_polarization(joinpath(OUT_DIR, "polarization.csv"), rows)
    sel = select_profile_voltages(sort(ok_Vs, by = v -> -v); nsel = 9)
    Vsel = sort(ok_Vs, by = v -> -v)[sel]
    samples = [(V, sols[V], fill(V - V_RHE_to_SHE_offset, clm.N), zeros(clm.N)) for V in Vsel]
    write_liquid_profiles(joinpath(OUT_DIR, "liquid_profiles.csv"), clm, samples)

    # ── physicality checks (§ Stage 1) ──
    println("\n── Physicality checks ──")
    @printf("Converged: %d/%d voltages (%d failed)\n", length(ok_Vs), length(Vs), nfail)
    # pH range across CL at a mid voltage
    Vmid = Vsel[cld(length(Vsel), 2)]
    cmid = unpack_log_conc(sols[Vmid], clm.N)
    pHmid = [-log10(cmid[1, ix] / 1000) for ix in 1:clm.N]
    @printf("pH across CL @ V_RHE=%.3f: [%.2f, %.2f]\n", Vmid, minimum(pHmid), maximum(pHmid))
    @printf("CO₂(aq) @ V_RHE=%.3f: GDL side=%.2f → electrolyte side=%.2f mol/m³\n",
            Vmid, cmid[5, 1], cmid[5, end])
    # current monotonic in |V|?
    itot = [r[5] for r in sort(rows, by = r -> -r[1])]
    dmono = sum(diff(itot) .< -1e-6)
    @printf("i_total sign-decreasing steps (mild→cathodic): %d\n", dmono)
    # peak FE_CO and its location
    fecos = [r[6] for r in rows]; ipk = argmax(fecos)
    @printf("peak FE_CO = %.1f%% at V_RHE=%.3f (exp peak 79.4%% near −1.205)\n",
            fecos[ipk], rows[ipk][1])
    # model currents at the 5 experimental voltages (nearest grid V)
    println("model vs Lees (pre-fit, no gas transport / potentials — mismatch expected):")
    for (k, Ve) in enumerate(E_exp_RHE)
        j = argmin(abs.([r[1] for r in rows] .- Ve))
        @printf("  V_RHE≈%.3f: i_CO=%.1f (exp %.1f), i_H2=%.1f (exp %.1f) mA/cm²\n",
                rows[j][1], rows[j][3], i_CO_exp[k], rows[j][4], i_H2_exp[k])
    end

    println("\nGenerating plots …")
    run(`python3 $(joinpath(HERE, "plot_results.py")) $(OUT_DIR) "Stage 1"`)

    @printf("\n=== Stage 1 complete: %d solutions cached in %s ===\n", length(ok_Vs), CACHE_DIR)
end

main()
