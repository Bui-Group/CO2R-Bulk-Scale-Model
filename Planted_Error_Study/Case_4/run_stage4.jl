# run_stage4.jl — Stage 4: finalize the Darcy pressure equation (advect=true).
# Same DOFs as Stage 3; pressure now SOLVED (continuity + convective gas flux). §6.2/§7.3.
# Self-anchoring continuation (§9.5.5): anchor at −0.300 V_RHE from equilibrium liquid +
# inlet gas at uniform p_total, tangent predictor down the grid. The Stage-3 cache is
# checked as a driver-order + mesh-consistency guard (Stage 4 is compared against Stage 3).

include(joinpath(@__DIR__, "co2r.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))

const OUT_DIR   = joinpath(HERE, "output_stage4")
const CACHE_DIR = joinpath(OUT_DIR, "cache")
const S3_CACHE  = joinpath(HERE, "output_stage3", "cache")
mkpath(CACHE_DIR)

function main()
    @printf("=== Stage 4 ===  (N_CL=%d, N_GDL=%d)\n", N_CL, N_GDL)
    @printf("i0_CO2R = %.1f A/m², λ_MHC = %.2f eV, HER_pf = %.3e, α_c = %.4f, k_GL = %.2e m/s\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER(), k_GL())
    isdir(S3_CACHE) && !isempty(readdir(S3_CACHE)) || error("Stage 3 cache missing — run Stage 3 first.")

    clm = make_cl_mesh(); gdlm = make_gdl_mesh(); gm = build_gas_mesh(gdlm, clm)
    c_eq = equilibrium_vec(1000.0, 1000.0)
    Ndof = 5*clm.N + 4*gm.Ng

    # ── §9.5 solver workspace (same liquid|gas pattern and weight blocks as Stage 3)
    prow, pcol = build_gas_pattern(clm.N, gm.N_GDL)
    Jt = pattern_matrix(prow, pcol, Ndof)
    plu = AffineLU()
    blocks = dof_blocks(4, clm.N, gm.N_GDL)
    n_log = 5*clm.N

    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    # cross-mesh cache guard: the Stage-4-vs-Stage-3 comparison below needs the same mesh
    u3, _ = load_nearest(S3_CACHE, "s3_", Vs[1])
    length(u3) == Ndof || error("Stage 3 cache DOF count $(length(u3)) ≠ Stage 4 Ndof $Ndof — mesh mismatch (rerun Stage 3 at this mesh).")

    # anchor guess (§9.5.5 table): equilibrium liquid + inlet ω / uniform p_total in every gas cell
    u_eq = zeros(5*clm.N); for ix in 1:clm.N; u_eq[5*(ix-1)+1:5*ix] .= log.(c_eq); end
    u = make_u0_stage3(u_eq, gm.Ng, clm.N)

    rows = Vector{NTuple{7,Float64}}()
    ok_Vs = Float64[]; sols = Dict{Float64,Vector{Float64}}()

    V_ref = Ref(Vs[1])
    res_closure!(F, u_) = residual_stage3!(F, u_, V_ref[], clm, gm, c_eq; advect = true)
    assemble! = make_ad_assembler(res_closure!, Ndof)

    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jt, n_log_dofs = n_log,
        on_converged = (V, uu, info) -> begin
            save_solution(CACHE_DIR, "s4_", copy(uu), V)
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

    write_polarization(joinpath(OUT_DIR, "polarization.csv"), rows)
    cp(joinpath(HERE, "output_stage3", "polarization.csv"),
       joinpath(OUT_DIR, "prev_polarization.csv"); force = true)
    sel = select_profile_voltages(sort(ok_Vs, by = v -> -v); nsel = 9)
    Vsel = sort(ok_Vs, by = v -> -v)[sel]
    liq_samples = [(V, sols[V], fill(V - V_RHE_to_SHE_offset, clm.N), zeros(clm.N)) for V in Vsel]
    write_liquid_profiles(joinpath(OUT_DIR, "liquid_profiles.csv"), clm, liq_samples)
    gas_samples = [(V, sols[V]) for V in Vsel]
    write_gas_profiles(joinpath(OUT_DIR, "gas_profiles.csv"), gm, clm.N, gas_samples)

    println("\n── Physicality checks ──")
    @printf("Converged: %d/%d (%d failed)\n", length(ok_Vs), length(Vs), nfail)
    Vhi = Vsel[end]; uhi = sols[Vhi]
    pcl = [gas_p(uhi, clm.N, g) for g in gm.N_GDL+1:gm.Ng]
    @printf("@ V_RHE=%.3f: pressure across CL = [%.3f, %.3f] Pa, ΔP_CL = %.4g Pa\n",
            Vhi, minimum(pcl), maximum(pcl), maximum(pcl) - minimum(pcl))
    pall = [gas_p(uhi, clm.N, g) for g in 1:gm.Ng]
    @printf("@ V_RHE=%.3f: total ΔP (inlet→CL) = %.4g Pa (should be small, O(1-100))\n",
            Vhi, maximum(pall) - minimum(pall))
    # Stage 4 vs Stage 3 current agreement at a mid voltage
    r3 = split.(readlines(joinpath(HERE, "output_stage3", "polarization.csv"))[2:end], ",")
    v3 = [parse(Float64, r[1]) for r in r3]; ico3 = [parse(Float64, r[3]) for r in r3]
    j = argmin(abs.([r[1] for r in rows] .- Vhi))
    j3 = argmin(abs.(v3 .- Vhi))
    @printf("i_CO Stage4 vs Stage3 @ V≈%.3f: %.2f vs %.2f mA/cm² (should be ~identical)\n",
            Vhi, rows[j][3], ico3[j3])
    itot = [r[5] for r in sort(rows, by = r -> -r[1])]
    @printf("i_total sign-decreasing steps: %d\n", sum(diff(itot) .< -1e-6))
    fecos = [r[6] for r in rows]; ipk = argmax(fecos)
    @printf("peak FE_CO = %.1f%% at V_RHE=%.3f\n", fecos[ipk], rows[ipk][1])

    println("\nGenerating plots …")
    withenv("PREV_LABEL" => "Stage 3") do
        run(`python3 $(joinpath(HERE, "plot_results.py")) $(OUT_DIR) "Stage 4" --gas`)
    end
    @printf("\n=== Stage 4 complete: %d solutions cached ===\n", length(ok_Vs))
end

main()
