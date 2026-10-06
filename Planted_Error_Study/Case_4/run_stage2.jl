# run_stage2.jl — Stage 2: Sherwood-Reynolds convective Neumann BC at the
# electrolyte side (§7.2). Same 5×N_CL DOFs. Self-anchoring continuation (§9.5.5):
# anchor at −0.300 V_RHE from the equilibrium guess, tangent predictor down the grid.
# The Stage-1 cache is only checked as a driver-order guard (§13).

include(joinpath(@__DIR__, "co2r.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))

const OUT_DIR   = joinpath(HERE, "output_stage2")
const CACHE_DIR = joinpath(OUT_DIR, "cache")
const S1_CACHE  = joinpath(HERE, "output_stage1", "cache")
mkpath(CACHE_DIR)

function main()
    @printf("=== Stage 2 ===  (N_CL=%d, N_GDL=%d)\n", N_CL, N_GDL)
    @printf("i0_CO2R = %.1f A/m², λ_MHC = %.2f eV, HER_pf = %.3e, α_c = %.4f\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER())
    isdir(S1_CACHE) && !isempty(readdir(S1_CACHE)) || error("Stage 1 cache missing — run Stage 1 first.")

    clm = make_cl_mesh()
    c_eq = equilibrium_vec(1000.0, 1000.0)
    c_CO2_bc = H_eff_vantHoff() * (x_inlet[1] * p_total / 101325.0)
    Ndof = 5 * clm.N

    # ── §9.5 solver workspace (fixed pattern, persistent LU, weight blocks)
    prow, pcol = block_tridiag_pattern(clm.N, 5)
    Jt = pattern_matrix(prow, pcol, Ndof)
    plu = AffineLU()
    blocks = dof_blocks(2, clm.N, N_GDL)

    # Biot numbers (physicality)
    Bi = [k_MT[i] * L_CL / D_eff[i] for i in 1:5]
    @printf("Biot numbers k_MT·L_CL/D_eff = [%.2f, %.2f, %.2f, %.2f, %.2f]\n", Bi...)

    u_eq = zeros(Ndof); for ix in 1:clm.N; u_eq[5*(ix-1)+1:5*ix] .= log.(c_eq); end
    u = copy(u_eq)                                     # anchor guess (§9.5.5 table)
    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    rows = Vector{NTuple{7,Float64}}()
    ok_Vs = Float64[]; sols = Dict{Float64,Vector{Float64}}()

    V_ref = Ref(Vs[1])
    res_closure!(F, u_) = residual_stage2!(F, u_, V_ref[], clm, c_CO2_bc, c_eq)
    assemble! = make_ad_assembler(res_closure!, Ndof)

    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jt, n_log_dofs = Ndof,
        on_converged = (V, uu, info) -> begin
            save_solution(CACHE_DIR, "s2_", copy(uu), V)
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
    cp(joinpath(HERE, "output_stage1", "polarization.csv"),
       joinpath(OUT_DIR, "prev_polarization.csv"); force = true)   # Stage 1 overlay
    sel = select_profile_voltages(sort(ok_Vs, by = v -> -v); nsel = 9)
    Vsel = sort(ok_Vs, by = v -> -v)[sel]
    samples = [(V, sols[V], fill(V - V_RHE_to_SHE_offset, clm.N), zeros(clm.N)) for V in Vsel]
    write_liquid_profiles(joinpath(OUT_DIR, "liquid_profiles.csv"), clm, samples)

    println("\n── Physicality checks ──")
    @printf("Converged: %d/%d (%d failed)\n", length(ok_Vs), length(Vs), nfail)
    # CO2(L_CL) depletion + OH(L_CL) vs bulk at a high-current voltage
    Vhi = Vsel[end]
    chi = unpack_log_conc(sols[Vhi], clm.N)
    @printf("@ V_RHE=%.3f: CO₂(aq) electrolyte face = %.3f mol/m³ (→0 if MT-limited)\n", Vhi, chi[5, end])
    @printf("@ V_RHE=%.3f: OH⁻ electrolyte face = %.4e mol/m³ (bulk eq = %.4e; should be above)\n",
            Vhi, chi[2, end], c_eq[2])
    # flux consistency at electrolyte face: J = k_MT (c_N − c_eq)
    Jco2 = -k_MT[5] * (c_eq[5] - chi[5, end])
    @printf("CO₂ flux @ electrolyte face = %.3e mol/m²/s (negative ⇒ reservoir resupplies)\n", Jco2)
    itot = [r[5] for r in sort(rows, by = r -> -r[1])]
    @printf("i_total sign-decreasing steps: %d\n", sum(diff(itot) .< -1e-6))
    fecos = [r[6] for r in rows]; ipk = argmax(fecos)
    @printf("peak FE_CO = %.1f%% at V_RHE=%.3f\n", fecos[ipk], rows[ipk][1])

    println("\nGenerating plots …")
    withenv("PREV_LABEL" => "Stage 1") do
        run(`python3 $(joinpath(HERE, "plot_results.py")) $(OUT_DIR) "Stage 2"`)
    end
    @printf("\n=== Stage 2 complete: %d solutions cached ===\n", length(ok_Vs))
end

main()
