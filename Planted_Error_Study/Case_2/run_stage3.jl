# ============================================================================
# run_stage3.jl — Stage 3 driver (§11): Stefan-Maxwell gas transport (u_G=0) +
# R_PT phase transfer + Sechenov Henry. Self-anchoring continuation (§9.5.5):
# anchor at −0.300 V_RHE from equilibrium liquid + inlet gas (= the gas-only
# solution) in every cell, tangent predictor down the grid. The Stage 2 cache is
# only checked as a driver-order guard.
# ============================================================================

const HERE = @__DIR__
include(joinpath(HERE, "params.jl"))
include(joinpath(HERE, "mesh.jl"))
include(joinpath(HERE, "chemistry.jl"))
include(joinpath(HERE, "kinetics.jl"))
include(joinpath(HERE, "transport.jl"))
include(joinpath(HERE, "gas_transport.jl"))
include(joinpath(HERE, "solver_sparse.jl"))
include(joinpath(HERE, "assembly.jl"))
include(joinpath(HERE, "cache.jl"))

const STAGE2_CACHE = joinpath(HERE, "output_stage2", "cache")
const STAGE3_DIR   = joinpath(HERE, "output_stage3")
const STAGE3_CACHE = joinpath(STAGE3_DIR, "cache")

# §9.5.5 anchor guess: bulk-equilibrium liquid in every CL cell, inlet gas
# composition and p_total in every gas cell.
function build_u0_stage3(ly, c_eq_vec)
    u0 = zeros(ndof_stage3(ly))
    ln = log.(c_eq_vec)
    for ix in 1:ly.N_CL; u0[liquid_idx(ix,1):liquid_idx(ix,5)] .= ln; end
    for cc in 1:(ly.N_GDL + ly.N_CL)
        b = _gas_base(ly, cc)
        u0[b+1] = omega_inlet[1]; u0[b+2] = omega_inlet[2]
        u0[b+3] = omega_inlet[3]; u0[b+4] = p_total
    end
    return u0
end

function run_stage3()
    println("="^70); println("=== Stage 3 ===")
    @printf("i0_CO2R = %.0f A/m², λ_MHC = %.2f eV\n", i0_CO2R(), lambda_MHC())
    @printf("HER_pf = %.3e, α_c = %.4f, k_GL = %.2e m/s\n", HER_prefactor(), alpha_c_HER(), k_GL())
    @printf("Mesh: N_CL=%d, N_GDL=%d (graded q=1.3)\n", N_CL, N_GDL)
    isdir(STAGE2_CACHE) && !isempty(readdir(STAGE2_CACHE)) || error("Run Stage 2 first")
    println("="^70)

    cl_mesh  = make_cl_mesh(N_CL, L_CL; q = 1.3)
    gdl_mesh = make_gdl_mesh(N_GDL, L_GDL; q = 1.3)
    gas_mesh = make_gas_mesh(gdl_mesh, cl_mesh)
    ly = GasLayout(N_CL, N_GDL)
    c_eq_vec = equilibrium_vec(1000.0, 1000.0)
    Ndof = ndof_stage3(ly)

    @printf("DOFs: %d (liquid %d + gas %d)\n", Ndof, 5*N_CL, 4*(N_GDL+N_CL))
    @printf("omega_inlet (mass) = [%.3f, %.3f, %.3f, %.3f]\n", omega_inlet...)
    @printf("D_bin (CO2-H2O) = %.3e m²/s\n", D_bin_raw[1,4])

    # §9.3.3 fixed pattern (liquid tridiag + liquid↔CL-gas + gas tridiag), §9.5.2 persistent LU
    rows, cols = build_gas_pattern(ly)
    Jtemplate  = pattern_matrix(rows, cols, Ndof)
    plu        = AffineLU()
    blocks     = dof_blocks(3; N_CL_ = ly.N_CL, N_GDL_ = ly.N_GDL)
    @assert blocks[2] == gas_resid_idx(ly, 1, 1):gas_resid_idx(ly, ly.N_GDL, 4)
    @assert blocks[3] == gas_resid_idx(ly, ly.N_GDL + 1, 1):gas_resid_idx(ly, ly.N_GDL + ly.N_CL, 4)
    Vs = sort(collect(V_RHE_grid), by = v -> -v)

    V_ref = Ref(Vs[1])
    res!  = (F_, u_) -> residual_stage3!(F_, u_, V_ref[], cl_mesh, gas_mesh, c_eq_vec, ly)
    assemble! = make_ad_assembler(res!, Ndof; chunk = 12)

    pol = NamedTuple[]; prof_rows = String[]; gas_rows = String[]
    u  = build_u0_stage3(ly, c_eq_vec)
    t0 = time()
    sweep_log = continuation_sweep!(u, Vs, res!, V_ref;
        blocks = blocks, plu = plu, jac_assembler = assemble!, J = Jtemplate, n_log_dofs = 5*N_CL,
        on_converged = (V_now, uu, info) -> begin
            save_solution(STAGE3_CACHE, "s3", copy(uu), V_now)

            V_app = V_now - V_RHE_to_SHE_offset
            iCO = compute_i_CO(uu, V_app, cl_mesh)/10.0; iH2 = compute_i_H2(uu, V_app, cl_mesh)/10.0
            itot = iCO+iH2; FEco = itot>0 ? 100*iCO/itot : 0.0
            push!(pol, (V_RHE=V_now, V_SHE=V_app, iCO=iCO, iH2=iH2, itot=itot,
                        FE_CO=FEco, FE_H2=100-FEco, iter=info.iterations, Finf=info.residual))

            c_all = unpack_log_conc(uu, N_CL); rco, rh2 = rate_profiles(uu, V_app, cl_mesh)
            for ix in 1:N_CL
                cv = @view c_all[:, ix]; pH = -log10(cv[1]/1000.0)
                push!(prof_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e",
                      V_now, cl_mesh.x_centers[ix]*1e6, pH, cv[1], cv[2], cv[3], cv[4], cv[5],
                      c_Cs_from_EN(cv), rco[ix], rh2[ix]))
            end
            for cc in 1:(N_GDL+N_CL)
                w = gas_cell_omega(uu, ly, cc); x = omega_to_x(w); p = gas_cell_p(uu, ly, cc)
                push!(gas_rows, @sprintf("%.6f,%.6f,%.8e,%.8e,%.8e,%.8e",
                      V_now, gas_mesh.x_centers[cc]*1e6, x[1]*p/101325, x[2]*p/101325,
                      x[3]*p/101325, p/101325))
            end
            @printf("V_RHE=%.3f: iCO=%7.2f iH2=%7.2f itot=%7.2f mA/cm²  FE_CO=%5.1f%%  iter=%d ‖F‖∞=%.1e\n",
                    V_now, iCO, iH2, itot, FEco, info.iterations, info.residual)
        end)
    nconv  = count(l -> l.converged, sweep_log)
    n_iter = sum(l.iterations for l in sweep_log)
    maxF   = maximum((l.residual for l in sweep_log if l.converged); init = 0.0)
    @printf("\nStage 3 sweep: %d/%d converged in %.1f s (%d Newton iterations, %.2f/step); max ‖F‖∞ = %.2e\n",
            nconv, length(Vs), time()-t0, n_iter, n_iter/length(Vs), maxF)

    mkpath(STAGE3_DIR)
    open(joinpath(STAGE3_DIR, "polarization.csv"), "w") do f
        println(f, "V_RHE,V_SHE,iCO_mAcm2,iH2_mAcm2,itot_mAcm2,FE_CO,FE_H2,iters,Finf")
        for p in sort(pol, by=r->r.V_RHE)
            @printf(f, "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d,%.3e\n",
                    p.V_RHE, p.V_SHE, p.iCO, p.iH2, p.itot, p.FE_CO, p.FE_H2, p.iter, p.Finf)
        end
    end
    open(joinpath(STAGE3_DIR, "profiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pH,c_H,c_OH,c_HCO3,c_CO3,c_CO2,c_Cs,R_CO2R,R_HER")
        for row in prof_rows; println(f, row); end
    end
    open(joinpath(STAGE3_DIR, "gasprofiles.csv"), "w") do f
        println(f, "V_RHE,x_um,pCO2_atm,pCO_atm,pH2_atm,ptot_atm")
        for row in gas_rows; println(f, row); end
    end

    # ── Physicality checks (§ Stage 3) ──
    println("\n--- Physicality checks ---")
    Vhi = Vs[end]
    uhi = load_solution(STAGE3_CACHE, "s3", Vhi)
    wlist = [gas_cell_omega(uhi, ly, cc) for cc in 1:(N_GDL+N_CL)]
    wco2_in  = wlist[1][1]; wco2_out = wlist[end][1]
    wco_out  = wlist[end][2]; wh2_out = wlist[end][3]
    @printf("1. ω_CO2 GDL-inlet=%.4f → CL-exit=%.4f @ V_RHE=%.3f (expect decrease)\n",
            wco2_in, wco2_out, Vhi)
    @printf("2. ω_CO CL-exit=%.4e, ω_H2 CL-exit=%.4e (expect increase from 0)\n", wco_out, wh2_out)
    # 3. Henry consistency at GDL/CL interface (first CL cell)
    c_all = unpack_log_conc(uhi, N_CL)
    pco2_if = pCO2_atm(uhi, ly, N_GDL+1)
    Heq = sechenov_H_CO2(@view c_all[:,1])
    @printf("3. Henry @ interface: H_eff=%.2f mol/(m³atm), p_CO2=%.4f atm → c_CO2*=%.3f vs c_CO2(aq,cell1)=%.3f\n",
            Heq, pco2_if, Heq*pco2_if, c_all[5,1])
    polV = sort(pol, by=r->-r.V_RHE)
    @printf("4. FE_CO peak: %.1f%% at V_RHE=%.3f\n",
            maximum(p->p.FE_CO, polV), polV[argmax([p.FE_CO for p in polV])].V_RHE)
    println("\nStage 3 complete. Plots: run plot_results.py 3")
    return nconv, length(Vs)
end

run_stage3()
