# ============================================================================
# run_stage1.jl — Stage 1 driver (§11): liquid NP diffusion-only + Dirichlet
# both sides + simplified kinetics (φ_s = V_applied_SHE, φ_L = 0).
# Solver: §9.5 stack (newton_affine! + persistent sparse LU + ForwardDiff AD)
# driven by the unified continuation of §9.5.5 — anchor at −0.300 V_RHE from the
# exact equilibrium guess, tangent predictor down the 89-V grid, §9.5.6 post-solve
# assertion at every emitted voltage. Caches every converged solution, exports
# diagnostics.
# ============================================================================

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "gas_transport.jl"))   # GasLayout type used by the Stage 3–5 residual signatures in assembly.jl
include(joinpath(HERE, "solver_sparse.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "cache.jl"))

const STAGE1_DIR   = joinpath(HERE, "output_stage1")
const STAGE1_CACHE = joinpath(STAGE1_DIR, "cache")
const PREFIX = "s1"

function build_equilibrium_u0(N::Int, c_eq_vec)
    ln_c_eq = log.(c_eq_vec)
    u0 = zeros(5 * N)
    for ix in 1:N
        u0[liquid_idx(ix, 1):liquid_idx(ix, 5)] .= ln_c_eq
    end
    return u0
end

function run_stage1()
    println("="^70)
    println("=== Stage 1 ===")
    @printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, α_c = %.4f\n", HER_prefactor(), alpha_c_HER())
    @printf("Mesh: N_CL = %d (graded q=1.3)\n", N_CL)
    @printf("Cache dir: %s\n", STAGE1_CACHE)
    if isdir(STAGE1_CACHE) && !isempty(readdir(STAGE1_CACHE))
        println("WARNING: Existing cache found. Verify kinetics match before reusing!")
    end
    println("="^70)

    cl_mesh  = make_cl_mesh(N_CL, L_CL; q = 1.3)
    c_eq_vec = equilibrium_vec(1000.0, 1000.0)
    eqnt     = solve_buffer_equilibrium(1000.0, 1000.0)
    @printf("Bulk equilibrium: pH=%.2f  c_CO2=%.3f  c_HCO3=%.1f  c_CO3=%.2f  c_OH=%.3e mol/m³\n",
            eqnt.pH, eqnt.CO2, eqnt.HCO3, eqnt.CO3, eqnt.OH)
    # Verify equilibrium satisfies all buffer rates (§9.6)
    Smax = maximum(abs, buffer_sources(c_eq_vec))
    @printf("max|buffer source| at equilibrium = %.3e mol/m³/s  (must be < 1e-6)\n", Smax)
    @assert Smax < 1e-6 "Buffer equilibrium inconsistent — check K values / unit conversions"

    # §9.3.3 fixed sparsity pattern (block-tridiagonal, 5×5 blocks) + §9.5.2 persistent LU
    Ndof       = 5 * N_CL
    rows, cols = block_tridiag_pattern(N_CL, 5)
    Jtemplate  = pattern_matrix(rows, cols, Ndof)
    plu        = AffineLU()
    blocks     = dof_blocks(1)

    Vs = sort(collect(V_RHE_grid), by = v -> -v)   # mild → cathodic

    # The residual captures V through V_ref, so one AD config/assembler serves the whole sweep.
    V_ref = Ref(Vs[1])
    res!  = (F_, u_) -> residual_stage1!(F_, u_, V_ref[], cl_mesh, c_eq_vec)
    assemble! = make_ad_assembler(res!, Ndof; chunk = 12)

    # Diagnostics accumulators
    pol = NamedTuple[]                              # polarization rows
    prof_rows = String[]                            # profile CSV rows

    # §9.5.5: anchor at −0.300 V from the equilibrium guess, tangent-predictor continuation down the grid
    u  = build_equilibrium_u0(N_CL, c_eq_vec)
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jtemplate, n_log_dofs = Ndof,
        on_converged = (V_now, uu, info) -> begin
            save_solution(STAGE1_CACHE, PREFIX, copy(uu), V_now)   # cache FIRST (§10.4)

            V_app = V_now - V_RHE_to_SHE_offset
            iCO = compute_i_CO(uu, V_app, cl_mesh) / 10.0    # A/m² → mA/cm²
            iH2 = compute_i_H2(uu, V_app, cl_mesh) / 10.0
            itot = iCO + iH2
            FEco = itot > 0 ? 100 * iCO / itot : 0.0
            push!(pol, (V_RHE = V_now, V_SHE = V_app, iCO = iCO, iH2 = iH2,
                        itot = itot, FE_CO = FEco, FE_H2 = 100 - FEco,
                        iter = info.iterations, Finf = info.residual))

            # Profile rows (all voltages)
            c_all = unpack_log_conc(uu, N_CL)
            rco, rh2 = rate_profiles(uu, V_app, cl_mesh)
            for ix in 1:N_CL
                cv = @view c_all[:, ix]
                cCs = c_Cs_from_EN(cv)
                pH = -log10(cv[1] / 1000.0)
                push!(prof_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                      V_now, cl_mesh.x_centers[ix]*1e6, pH, cv[1], cv[2], cv[3], cv[4], cv[5],
                      cCs, rco[ix], rh2[ix]))
            end

            @printf("V_RHE=%.3f (V_SHE=%.3f): iCO=%7.2f iH2=%7.2f itot=%7.2f mA/cm²  FE_CO=%5.1f%%  iter=%d ‖F‖∞=%.1e\n",
                    V_now, V_app, iCO, iH2, itot, FEco, info.iterations, info.residual)
        end)
    nconv  = count(l -> l.converged, sweep_log)
    n_iter = sum(l.iterations for l in sweep_log)
    maxF   = maximum((l.residual for l in sweep_log if l.converged); init = 0.0)
    @printf("\nStage 1 sweep: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step); max ‖F‖∞ = %.2e\n",
            nconv, length(Vs), time()-t0, n_iter, n_iter/length(Vs), maxF)

    # ── Export diagnostics ──
    mkpath(STAGE1_DIR)
    open(joinpath(STAGE1_DIR, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,iCO_mAcm2,iH2_mAcm2,itot_mAcm2,FE_CO,FE_H2,iters,Finf")
        for p in sort(pol, by = r -> r.V_RHE)
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d,%.3e\n",
                    p.V_RHE, p.V_SHE, p.iCO, p.iH2, p.itot, p.FE_CO, p.FE_H2, p.iter, p.Finf)
        end
    end
    open(joinpath(STAGE1_DIR, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_CO2,c_Cs,R_CO2R,R_HER")
        for row in prof_rows
            println(f, row)
        end
    end

    # ── Physicality checks (§11 Stage 1) ──
    println("\n--- Physicality checks ---")
    polS = sort(pol, by = r -> r.V_RHE)
    # 1. pH range across CL at a representative high-current voltage
    let
        u_hi = load_solution(STAGE1_CACHE, PREFIX, Vs[end])
        c_all = unpack_log_conc(u_hi, N_CL)
        pHs = [-log10(c_all[1, ix]/1000.0) for ix in 1:N_CL]
        @printf("1. pH range across CL @ V_RHE=%.3f: [%.2f, %.2f] (expect ~9–12)\n",
                Vs[end], minimum(pHs), maximum(pHs))
        @printf("2. CO2(aq) range @ same V: [%.3f, %.3f] mol/m³ (GDL face Dirichlet=%.2f)\n",
                minimum(c_all[5,:]), maximum(c_all[5,:]), c_CO2_henry_stage12)
    end
    # 3. monotonicity of total current vs |V| (order mild → cathodic)
    polV = sort(pol, by = r -> -r.V_RHE)            # mild (−0.3) first
    itots = [p.itot for p in polV]
    dmono = all(diff(itots) .>= -1e-6 .* maximum(abs, itots))
    @printf("3. i_total monotone increasing with |V|: %s\n", dmono ? "yes" : "NO")
    # 4. CO selectivity near onset (first voltage with appreciable current)
    onset = polV[findfirst(p -> p.itot > 5.0, polV)]
    @printf("4. FE_CO near onset (V_RHE=%.3f, itot=%.2f): %.1f%%  [guide check expects 80–95%%]\n",
            onset.V_RHE, onset.itot, onset.FE_CO)
    # 4b. FE_CO peak
    pk = polV[argmax([p.FE_CO for p in polV])]
    @printf("4b. FE_CO peak: %.1f%% at V_RHE=%.3f\n", pk.FE_CO, pk.V_RHE)

    println("\nStage 1 complete. Plots: run plot_results.py 1")
    return nconv, length(Vs)
end

run_stage1()
