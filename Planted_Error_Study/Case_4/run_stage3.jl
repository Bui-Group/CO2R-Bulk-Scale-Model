# run_stage3.jl — Stage 3: Stefan-Maxwell gas transport (u_G=0) + R_PT phase
# transfer. Adds 4·(N_GDL+N_CL) gas DOFs; pressure pinned to p_total. §6/§7.3/§7.4.
# Self-anchoring continuation (§9.5.5): anchor at −0.300 V_RHE from equilibrium
# liquid + inlet gas, tangent predictor down the grid. Stage-2 cache = driver-order guard.

include(joinpath(@__DIR__, "co2r.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))

const OUT_DIR   = joinpath(HERE, "output_stage3")
const CACHE_DIR = joinpath(OUT_DIR, "cache")
const S2_CACHE  = joinpath(HERE, "output_stage2", "cache")
mkpath(CACHE_DIR)

function main()
    @printf("=== Stage 3 ===  (N_CL=%d, N_GDL=%d)\n", N_CL, N_GDL)
    @printf("i0_CO2R = %.1f A/m², λ_MHC = %.2f eV, HER_pf = %.3e, α_c = %.4f, k_GL = %.2e m/s\n",
            i0_CO2R(), lambda_MHC(), HER_prefactor(), alpha_c_HER(), k_GL())
    isdir(S2_CACHE) && !isempty(readdir(S2_CACHE)) || error("Stage 2 cache missing — run Stage 2 first.")

    clm = make_cl_mesh(); gdlm = make_gdl_mesh(); gm = build_gas_mesh(gdlm, clm)
    c_eq = equilibrium_vec(1000.0, 1000.0)
    Ndof = 5*clm.N + 4*gm.Ng
    @printf("DOFs: %d (5·%d liquid + 4·%d gas)\n", Ndof, clm.N, gm.Ng)

    # ── §9.5 solver workspace: fixed liquid|gas structural pattern (§9.3.3), persistent LU
    #    (§9.5.2), weight blocks liquid | gas GDL | gas CL (§9.5.3)
    prow, pcol = build_gas_pattern(clm.N, gm.N_GDL)
    Jt = pattern_matrix(prow, pcol, Ndof)
    plu = AffineLU()
    blocks = dof_blocks(3, clm.N, gm.N_GDL)
    n_log = 5*clm.N                                            # clamp only liquid log-DOFs

    # anchor guess (§9.5.5 table): equilibrium liquid + inlet ω / p_total in every gas cell
    u_eq = zeros(5*clm.N); for ix in 1:clm.N; u_eq[5*(ix-1)+1:5*ix] .= log.(c_eq); end
    u = make_u0_stage3(u_eq, gm.Ng, clm.N)

    Vs = sort(collect(V_RHE_grid), by = v -> -v)
    rows = Vector{NTuple{7,Float64}}()
    ok_Vs = Float64[]; sols = Dict{Float64,Vector{Float64}}()

    V_ref = Ref(Vs[1])
    res_closure!(F, u_) = residual_stage3!(F, u_, V_ref[], clm, gm, c_eq; advect = false)
    assemble! = make_ad_assembler(res_closure!, Ndof)

    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res_closure!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jt, n_log_dofs = n_log,
        on_converged = (V, uu, info) -> begin
            save_solution(CACHE_DIR, "s3_", copy(uu), V)
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
    cp(joinpath(HERE, "output_stage2", "polarization.csv"),
       joinpath(OUT_DIR, "prev_polarization.csv"); force = true)
    sel = select_profile_voltages(sort(ok_Vs, by = v -> -v); nsel = 9)
    Vsel = sort(ok_Vs, by = v -> -v)[sel]
    liq_samples = [(V, sols[V], fill(V - V_RHE_to_SHE_offset, clm.N), zeros(clm.N)) for V in Vsel]
    write_liquid_profiles(joinpath(OUT_DIR, "liquid_profiles.csv"), clm, liq_samples)
    gas_samples = [(V, sols[V]) for V in Vsel]
    write_gas_profiles(joinpath(OUT_DIR, "gas_profiles.csv"), gm, clm.N, gas_samples)

    println("\n── Physicality checks ──")
    @printf("Converged: %d/%d (%d failed)\n", length(ok_Vs), length(Vs), nfail)
    Vhi = Vsel[end]
    uhi = sols[Vhi]
    ω_gdl_in = gas_omega(uhi, clm.N, 1); ω_cl_out = gas_omega(uhi, clm.N, gm.Ng)
    @printf("@ V_RHE=%.3f: ω_CO₂ GDL inlet=%.4f → CL exit=%.4f (should decrease)\n",
            Vhi, ω_gdl_in[1], ω_cl_out[1])
    @printf("@ V_RHE=%.3f: ω_CO CL exit=%.4e, ω_H₂ CL exit=%.4e (should be >0)\n",
            Vhi, ω_cl_out[2], ω_cl_out[3])
    xco2 = omega_to_x(ω_cl_out)[1]
    @printf("@ V_RHE=%.3f: x_CO₂ CL exit=%.4f, p_CO₂=%.4f atm\n", Vhi, xco2, xco2*gas_p(uhi,clm.N,gm.Ng)/101325)
    pminmax = extrema([gas_p(uhi, clm.N, g) for g in 1:gm.Ng])
    @printf("pressure range = [%.1f, %.1f] Pa (pinned ⇒ should be p_total=%.1f)\n", pminmax..., p_total)
    itot = [r[5] for r in sort(rows, by = r -> -r[1])]
    @printf("i_total sign-decreasing steps: %d\n", sum(diff(itot) .< -1e-6))
    fecos = [r[6] for r in rows]; ipk = argmax(fecos)
    @printf("peak FE_CO = %.1f%% at V_RHE=%.3f\n", fecos[ipk], rows[ipk][1])

    println("\nGenerating plots …")
    withenv("PREV_LABEL" => "Stage 2") do
        run(`python3 $(joinpath(HERE, "plot_results.py")) $(OUT_DIR) "Stage 3" --gas`)
    end
    @printf("\n=== Stage 3 complete: %d solutions cached ===\n", length(ok_Vs))
end

main()
